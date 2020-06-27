import logging
from typing import Text, Any, List, Dict, Tuple, TYPE_CHECKING, Union

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.data import YAML_FILE_EXTENSIONS
from rasa.nlu.training_data.entities_parser import EntitiesParser
from rasa.nlu.training_data.formats.readerwriter import TrainingDataReader
import rasa.utils.io as io_utils
from rasa.nlu.training_data.lookup_tables_parser import LookupTablesParser
from rasa.nlu.training_data.synonyms_parser import SynonymsParser
from rasa.utils.common import raise_warning

if TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData


logger = logging.getLogger(__name__)

KEY_NLU = "nlu"
KEY_INTENT = "intent"
KEY_INTENT_EXAMPLES = "examples"
KEY_INTENT_TEXT = "text"
KEY_SYNONYM = "synonym"
KEY_SYNONYM_EXAMPLES = "examples"
KEY_REGEX = "regex"
KEY_REGEX_EXAMPLES = "examples"
KEY_LOOKUP = "lookup"
KEY_LOOKUP_EXAMPLES = "examples"


class RasaYAMLReader(TrainingDataReader):
    """Reads YAML training data and creates a TrainingData object."""

    def _initialize(self) -> None:
        self.training_examples = []
        self.entity_synonyms = {}
        self.regex_features = []
        self.lookup_tables = []

    def reads(self, string: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData in YAML format from a string

        Args:
            string: string with YAML training data
            **kwargs: key-value args

        Returns:
            New `TrainingData` object with parsed training data.
        """
        from rasa.nlu.training_data import TrainingData

        self._initialize()

        yaml_content = io_utils.read_yaml(string)

        for key, value in yaml_content.items():  # pytype: disable=attribute-error
            if key == KEY_NLU:
                self._parse_nlu(value)

        return TrainingData(
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        )

    def _parse_nlu(self, nlu_data: List[Dict[Text, Any]]) -> None:

        for nlu_item in nlu_data:
            if not isinstance(nlu_item, dict):
                raise_warning(
                    f"Unexpected block found in '{self.filename}':\n"
                    f"{nlu_item}\n"
                    f"Items under the `{KEY_NLU}` key must be YAML dictionaries. "
                    f"This block will be skipped. ",
                    docs=DOCS_URL_TRAINING_DATA_NLU,
                )
                continue

            for key in nlu_item.keys():
                if key == KEY_INTENT:
                    self._parse_intent(nlu_item)
                elif key == KEY_SYNONYM:
                    self._parse_synonym(nlu_item)
                elif key == KEY_REGEX:
                    self._parse_regex(nlu_item)
                elif key == KEY_LOOKUP:
                    self._parse_lookup(nlu_item)
                else:
                    raise_warning(
                        f"Unexpected key `{key}` found in '{self.filename}'."
                        f"Supported keys are: `{KEY_INTENT}`, `{KEY_SYNONYM}`, "
                        f"`{KEY_REGEX}`, `{KEY_LOOKUP}`. ",
                        docs=DOCS_URL_TRAINING_DATA_NLU,
                    )

    def _parse_intent(self, data: Dict[Text, Any]) -> None:
        from rasa.nlu.training_data import Message

        intent = data.get(KEY_INTENT, "")
        if not intent:
            raise_warning(
                f"Issue found while processing `{self.filename}`:"
                f"The intent has an empty name."
                f"Intents should have a name defined under the {KEY_INTENT}"
                "It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = data.get(KEY_INTENT_EXAMPLES, "")

        for example, entities in self._parse_training_examples(examples, intent):

            SynonymsParser.add_synonyms_from_entities(
                example, entities, self.entity_synonyms
            )

            plain_text = EntitiesParser.replace_entities(example)

            message = Message.build(plain_text, intent)
            if entities:
                message.set("entities", entities)
            self.training_examples.append(message)

    def _parse_training_examples(
        self, examples: Union[Text, List[Text]], intent: Text
    ) -> List[Tuple[Text, List[Dict]]]:

        if isinstance(examples, list):
            example_strings = [
                example.get(KEY_INTENT_TEXT, "")  # pytype: disable=attribute-error
                for example in examples
                if example
            ]
        elif isinstance(examples, str):
            example_strings = examples.splitlines()
        else:
            raise_warning(
                f"Unexpected block found in '{self.filename}' "
                f"while processing intent `{intent}`:\n"
                f"{examples}\n"
                f"This block will be skipped. ",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return []

        results = []
        for example in example_strings:
            entities = EntitiesParser.find_entities_in_training_example(example)
            results.append((example, entities))

        return results

    def _parse_synonym(self, nlu_item: Dict[Text, Any]) -> None:

        synonym_name = nlu_item[KEY_SYNONYM]
        if not synonym_name:
            raise_warning(
                f"Issue found while processing `{self.filename}`: "
                f"The synonym has an empty name. "
                f"Synonyms should have a name defined under the {KEY_SYNONYM}. "
                "It will be skipped. ",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_SYNONYM_EXAMPLES, "")

        if not examples:
            raise_warning(
                f"Issue found while processing `{self.filename}`: "
                f"{KEY_SYNONYM}: {synonym_name} doesn't have any examples. "
                "It will be skipped. ",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in {self.filename}:\n"
                f">> {examples}\n"
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in examples.splitlines():
            SynonymsParser.add_synonym(example, synonym_name, self.entity_synonyms)

    def _parse_regex(self, nlu_item: Dict[Text, Any]) -> None:

        regex_name = nlu_item[KEY_REGEX]
        if not regex_name:
            raise_warning(
                f"Issue found while processing `{self.filename}`:"
                f"The regex has an empty name."
                f"Regex should have a name defined under the {KEY_REGEX}"
                "It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_REGEX_EXAMPLES, "")
        if not examples:
            raise_warning(
                f"Issue found while processing `{self.filename}`:"
                f"{KEY_REGEX}: {regex_name} doesn't have any examples."
                "It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in {self.filename}:\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in examples.splitlines():
            self.regex_features.append({"name": regex_name, "pattern": example})

    def _parse_lookup(self, nlu_item: Dict[Text, Any]):

        lookup_item_name = nlu_item[KEY_LOOKUP]
        if not lookup_item_name:
            raise_warning(
                f"Issue found while processing `{self.filename}`:"
                f"The lookup item has an empty name."
                f"Lookup items should have a name defined under the {KEY_LOOKUP}"
                "It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_LOOKUP_EXAMPLES, "")
        if not examples:
            raise_warning(
                f"Issue found while processing `{self.filename}`:"
                f"{KEY_LOOKUP}: {lookup_item_name} doesn't have any examples."
                "It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in {self.filename}:\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in examples.splitlines():
            LookupTablesParser.add_item_to_lookup_tables(
                lookup_item_name, example, self.lookup_tables
            )

    @staticmethod
    def is_yaml_nlu_file(filename: Text) -> bool:
        """Checks if the specified file possibly contains NLU training data in YAML.

        Args:
            filename: name of the file to check.

        Returns:
            `True` if the `filename` is possibly a valid YAML MLU file,
            `False` otherwise.
        """
        if not filename.split(".")[-1] in YAML_FILE_EXTENSIONS:
            return False
        try:
            content = io_utils.read_yaml_file(filename)
            if KEY_NLU in content:
                return True
        # Using broad Exception because YAML lib doesn't expose required Errors
        except Exception as e:
            logger.error(
                f"Tried to check if '{filename}' is a NLU file, but failed to "
                f"read it. If this file contains story data, you should "
                f"investigate this error, otherwise it is probably best to "
                f"move the file to a different location. "
                f"Error: {e}"
            )
        return False
