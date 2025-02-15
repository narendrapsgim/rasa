from __future__ import annotations
from collections import defaultdict
from typing import Iterable, List, Dict, Text, Any, Set, Type

from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.featurizers.featurizer import Featurizer2
from rasa.nlu.extractors.mitie_entity_extractor import (
    MitieEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.regex_entity_extractor import (
    RegexEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.crf_entity_extractor import (
    CRFEntityExtractorGraphComponent,
    CRFEntityExtractorOptions,
)
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapperComponent
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import (
    RegexFeaturizerGraphComponent,
)
from rasa.nlu.classifiers.diet_classifier import DIETClassifierGraphComponent
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from rasa.nlu.tokenizers.tokenizer import TokenizerGraphComponent
from rasa.core.policies.rule_policy import RulePolicyGraphComponent
from rasa.core.policies.policy import PolicyGraphComponent, SupportedData
from rasa.core.policies.memoization import MemoizationPolicyGraphComponent
from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
from rasa.shared.core.training_data.structures import RuleStep, StoryGraph
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DOCS_URL_COMPONENTS,
    DOCS_URL_DEFAULT_ACTIONS,
    DOCS_URL_POLICIES,
    DOCS_URL_RULES,
)
from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.constants import (
    ACTION_BACK_NAME,
    ACTION_RESTART_NAME,
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
)
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.io


# TODO: replace these once the Recipe is merged
TRAINABLE_EXTRACTORS = [
    MitieEntityExtractorGraphComponent,
    CRFEntityExtractorGraphComponent,
    DIETClassifierGraphComponent,
]
# TODO: replace these once the Recipe is merged (used in tests)
POLICY_CLASSSES = {
    TEDPolicyGraphComponent,
    MemoizationPolicyGraphComponent,
    RulePolicyGraphComponent,
}


def _types_to_str(types: Iterable[Type]) -> Text:
    """Returns a text containing the names of all given types.

    Args:
        types: some types
    Returns:
        text containing all type names
    """
    return ", ".join([type.__name__ for type in types])


class DefaultV1RecipeValidator(GraphComponent):
    """Validates a "DefaultV1" configuration against the training data and domain."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DefaultV1RecipeValidator:
        """Creates a new `ConfigValidator` (see parent class for full docstring)."""
        return cls(execution_context.graph_schema)

    def __init__(self, graph_schema: GraphSchema) -> None:
        """Instantiates a new `ConfigValidator`.

        Args:
           graph_schema: a graph schema
        """
        self._graph_schema = graph_schema
        self._component_types = set(node.uses for node in graph_schema.nodes.values())
        self._policy_schema_nodes: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if issubclass(node.uses, PolicyGraphComponent)
        ]

    def validate(self, importer: TrainingDataImporter) -> TrainingDataImporter:
        """Validates the current graph schema against the training data and domain.

        Args:
            importer: the training data importer which can also load the domain
        Raises:
            `InvalidConfigException` or `InvalidDomain` in case there is some mismatch
        """
        nlu_data = importer.get_nlu_data()
        self._validate_nlu(nlu_data)

        story_graph = importer.get_stories()
        domain = importer.get_domain()
        self._validate_core(story_graph, domain)
        return importer

    def _validate_nlu(self, training_data: TrainingData) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           training_data: The training data for the NLU components.
        """
        self._raise_if_more_than_one_tokenizer()
        self._raise_if_featurizers_are_not_compatible()
        self._warn_of_competing_extractors()
        self._warn_of_competition_with_regex_extractor(training_data=training_data)
        self._warn_if_some_training_data_is_unused(training_data=training_data)

    def _warn_if_some_training_data_is_unused(
        self, training_data: TrainingData
    ) -> None:
        """Validates that all training data will be consumed by some component.

        For example, if you specify response examples in your training data, but there
        is no `ResponseSelector` component in your configuration, then this method
        issues a warning.

        Args:
            training_data: The training data for the NLU components.
        """
        if (
            training_data.response_examples
            and ResponseSelectorGraphComponent not in self._component_types
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data with examples for training a response "
                f"selector, but your NLU configuration does not include a response "
                f"selector component. "
                f"To train a model on your response selector data, add a "
                f"'{ResponseSelectorGraphComponent.__name__}' to your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.entity_examples and self._component_types.isdisjoint(
            TRAINABLE_EXTRACTORS
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data consisting of entity examples, but "
                f"your NLU configuration does not include an entity extractor "
                f"trained on your training data. "
                f"To extract non-pretrained entities, add one of "
                f"{_types_to_str(TRAINABLE_EXTRACTORS)} to your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.entity_examples and self._component_types.isdisjoint(
            {DIETClassifierGraphComponent, CRFEntityExtractorGraphComponent}
        ):
            if training_data.entity_roles_groups_used():
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data with entities that "
                    f"have roles/groups, but your NLU configuration does not "
                    f"include a '{DIETClassifierGraphComponent.__name__}' "
                    f"or a '{CRFEntityExtractorGraphComponent.__name__}'. "
                    f"To train entities that have roles/groups, "
                    f"add either '{DIETClassifierGraphComponent.__name__}' "
                    f"or '{CRFEntityExtractorGraphComponent.__name__}' to your "
                    f"configuration.",
                    docs=DOCS_URL_COMPONENTS,
                )

        if training_data.regex_features and self._component_types.isdisjoint(
            [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data with regexes, but "
                f"your NLU configuration does not include a 'RegexFeaturizer' "
                f" or a "
                f"'RegexEntityExtractor'. To use regexes, include either a "
                f"'{RegexFeaturizerGraphComponent.__name__}' or a "
                f"'{RegexEntityExtractorGraphComponent.__name__}' "
                f"in your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.lookup_tables and self._component_types.isdisjoint(
            [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data consisting of lookup tables, but "
                f"your NLU configuration does not include a featurizer "
                f"or an entity extractor using the lookup table."
                f"To use the lookup tables, include either a "
                f"'{RegexFeaturizerGraphComponent.__name__}' "
                f"or a '{RegexEntityExtractorGraphComponent.__name__}' "
                f"in your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.lookup_tables:

            if self._component_types.isdisjoint(
                [CRFEntityExtractorGraphComponent, DIETClassifierGraphComponent]
            ):
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data consisting of lookup tables, but "
                    f"your NLU configuration does not include any components "
                    f"that uses the features created from the lookup table. "
                    f"To make use of the features that are created with the "
                    f"help of the lookup tables, "
                    f"add a '{DIETClassifierGraphComponent.__name__}' or a "
                    f"'{CRFEntityExtractorGraphComponent.__name__}' "
                    f"with the 'pattern' feature "
                    f"to your configuration.",
                    docs=DOCS_URL_COMPONENTS,
                )

            elif CRFEntityExtractorGraphComponent in self._component_types:

                crf_schema_nodes = [
                    schema_node
                    for schema_node in self._graph_schema.nodes.values()
                    if schema_node.uses == CRFEntityExtractorGraphComponent
                ]
                has_pattern_feature = any(
                    CRFEntityExtractorOptions.PATTERN in feature_list
                    for crf in crf_schema_nodes
                    for feature_list in crf.config.get("features", [])
                )

                if not has_pattern_feature:
                    rasa.shared.utils.io.raise_warning(
                        f"You have defined training data consisting of "
                        f"lookup tables, but your NLU configuration's "
                        f"'{CRFEntityExtractorGraphComponent.__name__}' "
                        f"does not include the "
                        f"'pattern' feature. To featurize lookup tables, "
                        f"add the 'pattern' feature to the "
                        f"'{CRFEntityExtractorGraphComponent.__name__}' "
                        "in your configuration.",
                        docs=DOCS_URL_COMPONENTS,
                    )

        if (
            training_data.entity_synonyms
            and EntitySynonymMapperComponent not in self._component_types
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined synonyms in your training data, but "
                f"your NLU configuration does not include an "
                f"'{EntitySynonymMapperComponent.__name__}'. "
                f"To map synonyms, add an "
                f"'{EntitySynonymMapperComponent.__name__}' to your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

    def _raise_if_more_than_one_tokenizer(self) -> None:
        """Validates that only one tokenizer is present in the configuration.

        Note that the existence of a tokenizer and its position in the graph schema
        will be validated via the validation of required components during
        schema validation.

        Raises:
            `InvalidConfigException` in case there is more than one tokenizer
        """
        types_of_tokenizer_schema_nodes = [
            schema_node.uses
            for schema_node in self._graph_schema.nodes.values()
            if issubclass(schema_node.uses, TokenizerGraphComponent)
        ]

        if len(types_of_tokenizer_schema_nodes) > 1:
            raise InvalidConfigException(
                f"The configuration configuration contains more than one tokenizer, "
                f"which is not possible at this time. You can only use one tokenizer. "
                f"The configuration contains the following tokenizers: "
                f"{_types_to_str(types_of_tokenizer_schema_nodes)}. "
            )

    def _warn_of_competing_extractors(self) -> None:
        """Warns the user when using competing extractors.

        Competing extractors are e.g. `CRFEntityExtractor` and `DIETClassifier`.
        Both of these look for the same entities based on the same training data
        leading to ambiguity in the results.

        Args:
           node_names: names of all components
        """
        extractors_in_configuration: Set[
            Type[GraphComponent]
        ] = self._component_types.intersection(TRAINABLE_EXTRACTORS)
        if len(extractors_in_configuration) > 1:
            rasa.shared.utils.io.raise_warning(
                f"You have defined multiple entity extractors that do the same job "
                f"in your configuration: "
                f"{_types_to_str(extractors_in_configuration)}. "
                f"This can lead to the same entity getting "
                f"extracted multiple times. Please read the documentation section "
                f"on entity extractors to make sure you understand the implications.",
                docs=f"{DOCS_URL_COMPONENTS}#entity-extractors",
            )

    def _warn_of_competition_with_regex_extractor(
        self, training_data: TrainingData
    ) -> None:
        """Warns when regex entity extractor is competing with a general one.

        This might be the case when the following conditions are all met:
        * You are using a general entity extractor and the `RegexEntityExtractor`
        * AND you have regex patterns for entity type A
        * AND you have annotated text examples for entity type A

        Args:
            training_data: The training data for the NLU components.
        """
        present_general_extractors = self._component_types.intersection(
            TRAINABLE_EXTRACTORS
        )
        has_general_extractors = len(present_general_extractors) > 0
        has_regex_extractor = (
            RegexEntityExtractorGraphComponent in self._component_types
        )

        regex_entity_types = {rf["name"] for rf in training_data.regex_features}
        overlap_between_types = training_data.entities.intersection(regex_entity_types)
        has_overlap = len(overlap_between_types) > 0

        if has_general_extractors and has_regex_extractor and has_overlap:
            rasa.shared.utils.io.raise_warning(
                f"You have an overlap between the "
                f"'{RegexEntityExtractorGraphComponent.__name__}' and the "
                f"statistical entity extractors "
                f"{_types_to_str(present_general_extractors)} "
                f"in your configuration. Specifically both types of extractors will "
                f"attempt to extract entities of the types "
                f"{', '.join(overlap_between_types)}. "
                f"This can lead to multiple "
                f"extraction of entities. Please read "
                f"'{RegexEntityExtractorGraphComponent.__name__}''s "
                f"documentation section to make sure you understand the "
                f"implications.",
                docs=f"{DOCS_URL_COMPONENTS}#regexentityextractor",
            )

    def _raise_if_featurizers_are_not_compatible(self,) -> None:
        """Raises or warns if there are problems regarding the featurizers.

        Raises:
            `InvalidConfigException` in case the featurizers are not compatible
        """
        featurizers: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if issubclass(node.uses, Featurizer2)
        ]
        Featurizer2.raise_if_featurizer_configs_are_not_compatible(
            [schema_node.config for schema_node in featurizers]
        )

    def _validate_core(self, story_graph: StoryGraph, domain: Domain) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           story_graph: a story graph (core training data)
           domain: the domain
        """
        if not self._policy_schema_nodes and story_graph.story_steps:
            rasa.shared.utils.io.raise_warning(
                "Found data for training policies but no policy was configured.",
                docs=DOCS_URL_POLICIES,
            )
        if not self._policy_schema_nodes:
            return
        self._warn_if_no_rule_policy_is_contained()
        self._raise_if_domain_contains_form_names_but_no_rule_policy_given(domain)
        self._raise_if_a_rule_policy_is_incompatible_with_domain(domain)
        self._warn_if_priorities_are_not_unique()
        self._warn_if_rule_based_data_is_unused_or_missing(story_graph=story_graph)

    def _warn_if_no_rule_policy_is_contained(self) -> None:
        """Warns if there is no rule policy among the given policies."""
        if not any(
            node.uses == RulePolicyGraphComponent for node in self._policy_schema_nodes
        ):
            rasa.shared.utils.io.raise_warning(
                f"'{RulePolicyGraphComponent.__name__}' is not included in the model's "
                f"policy configuration. Default intents such as "
                f"'{USER_INTENT_RESTART}' and '{USER_INTENT_BACK}' will "
                f"not trigger actions '{ACTION_RESTART_NAME}' and "
                f"'{ACTION_BACK_NAME}'.",
                docs=DOCS_URL_DEFAULT_ACTIONS,
            )

    def _raise_if_domain_contains_form_names_but_no_rule_policy_given(
        self, domain: Domain,
    ) -> None:
        """Validates that there exists a rule policy if forms are defined.

        Raises:
            `InvalidConfigException` if domain and rule policies do not match
        """
        contains_rule_policy = any(
            schema_node
            for schema_node in self._graph_schema.nodes.values()
            if schema_node.uses == RulePolicyGraphComponent
        )

        if domain.form_names and not contains_rule_policy:
            raise InvalidDomain(
                "You have defined a form action, but have not added the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy ensemble. "
                f"Either remove all forms from your domain or add the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy configuration."
            )

    def _raise_if_a_rule_policy_is_incompatible_with_domain(
        self, domain: Domain,
    ) -> None:
        """Validates the rule policies against the domain.

        Raises:
            `InvalidDomain` if domain and rule policies do not match
        """
        for schema_node in self._graph_schema.nodes.values():
            if schema_node.uses == RulePolicyGraphComponent:
                RulePolicyGraphComponent.raise_if_incompatible_with_domain(
                    config=schema_node.config, domain=domain
                )

    def _warn_if_priorities_are_not_unique(self) -> None:
        """Warns if the priorities of the policies are not unique."""
        priority_dict = defaultdict(list)
        for schema_node in self._policy_schema_nodes:
            default_priority = schema_node.uses.get_default_config()["priority"]
            priority = schema_node.config.get("priority", default_priority)
            priority_dict[priority].append(schema_node.uses)

        for k, v in priority_dict.items():
            if len(v) > 1:
                rasa.shared.utils.io.raise_warning(
                    f"Found policies {_types_to_str(v)} with same priority {k} "
                    f"in PolicyEnsemble. When personalizing "
                    f"priorities, be sure to give all policies "
                    f"different priorities.",
                    docs=DOCS_URL_POLICIES,
                )

    def _warn_if_rule_based_data_is_unused_or_missing(
        self, story_graph: StoryGraph
    ) -> None:
        """Warns if rule-data is unused or missing.

        Args:
            story_graph: a story graph (core training data)
        """
        consuming_rule_data = any(
            policy_node.uses.supported_data()
            in [SupportedData.RULE_DATA, SupportedData.ML_AND_RULE_DATA]
            for policy_node in self._policy_schema_nodes
        )

        # Reminder: We generate rule trackers via:
        #   rasa/shared/core/generator/...
        #   .../TrainingDataGenerator/_generate_rule_trackers/
        contains_rule_tracker = any(
            isinstance(step, RuleStep) for step in story_graph.ordered_steps()
        )

        if consuming_rule_data and not contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found a rule-based policy in your configuration but "
                f"no rule-based training data. Please add rule-based "
                f"stories to your training data or "
                f"remove the rule-based policy "
                f"(`{RulePolicyGraphComponent.__name__}`) from your "
                f"your configuration.",
                docs=DOCS_URL_RULES,
            )
        elif not consuming_rule_data and contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found rule-based training data but no policy supporting rule-based "
                f"data. Please add `{RulePolicyGraphComponent.__name__}` "
                f"or another rule-supporting "
                f"policy to the `policies` section in `{DEFAULT_CONFIG_PATH}`.",
                docs=DOCS_URL_RULES,
            )
