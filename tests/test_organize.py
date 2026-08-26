"""
White-box tests for fuzzy_ml.organize.SelfOrganize.

SelfOrganize always adds fetch_labeled_dataset and KnowledgeBase.create to its
algorithm set automatically (see SelfOrganize.__init__) - both are plain
functions, not Node instances, so Regime.get_regime_edges()'s Node-specific
edge-wiring loop never references fetch_labeled_dataset at all, and only
wires KnowledgeBase.create in when no other algorithm already produces a
"knowledge_base" resource. In both cases here, neither ends up connected to
anything and Regime's own clean_up=True removes them as isolated vertices
before Regime.start() ever runs - confirmed directly (not assumed) by
inspecting get_regime_edges()'s output and by running SelfOrganize.run() end
to end with a dummy Node standing in for a real (CLIP-produced) knowledge
base, which frequent_discernible()/clip_frequent_discernible() themselves
cannot currently exercise (see tests/test_fyd/test_heuristic.py's module
docstring for why).

run()'s "found under KnowledgeBase.create's own module.name key" branch
(the default path, vs. a Node like FTARM producing "knowledge_base"
directly) needs a real LinguisticVariables placed inside a regime.Resource -
a namedtuple, so it must go into a Python set(), which requires
LinguisticVariables to be hashable. fuzzy-theory's LinguisticVariables is a
plain @dataclass, and a plain @dataclass's auto-generated __eq__ disables
__hash__ by default - fixed upstream (fuzzy-theory commit `03c1015`,
`@dataclass(eq=False)`) rather than worked around here.
"""

import unittest
import unittest.mock

import torch
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from regime import Node, Resource, hyperparameter

from fuzzy_ml.organize import SelfOrganize

AVAILABLE_DEVICE = torch.device("cpu")


class DummyKnowledgeBaseProducer(Node):
    """
    Stands in for a real CLIP/ECM/WangMendel -> KnowledgeBase.create pipeline:
    a Node whose resource_name is "knowledge_base", so SelfOrganize treats it
    as already producing the knowledge base and skips wiring in the real
    KnowledgeBase.create.
    """

    def __init__(self, resource_name: str = "knowledge_base"):
        super().__init__(resource_name)

    @hyperparameter("scale")
    def __call__(self, seed_value: float, scale: float, device: torch.device):
        return seed_value * scale


class TestSelfOrganizeInit(unittest.TestCase):
    """
    Covers SelfOrganize.__init__/__getitem__'s bookkeeping.
    """

    def test_always_adds_the_two_helper_functions(self) -> None:
        """
        fetch_labeled_dataset and KnowledgeBase.create are added automatically
        alongside whatever algorithms were explicitly provided.
        """
        node = DummyKnowledgeBaseProducer()
        self_organize = SelfOrganize(algorithms={"kb": node}, device=AVAILABLE_DEVICE)
        self.assertEqual(len(self_organize.algorithms), 3)
        self.assertIn(node, self_organize.algorithms)

    def test_getitem_returns_the_algorithm_by_its_dict_key(self) -> None:
        """
        __getitem__ looks up by the original dict key (e.g. "kb"), not by
        resource_name or any other attribute.
        """
        node = DummyKnowledgeBaseProducer()
        self_organize = SelfOrganize(algorithms={"kb": node}, device=AVAILABLE_DEVICE)
        self.assertIs(self_organize["kb"], node)


class TestGetRegimeEdges(unittest.TestCase):
    """
    Covers SelfOrganize.get_regime_edges()'s two branches.
    """

    def test_skips_knowledge_base_create_when_a_node_already_produces_it(
        self,
    ) -> None:
        """
        When an algorithm's resource_name is "knowledge_base", no edges are
        added for the real KnowledgeBase.create - it's left as an isolated,
        unused vertex (removed later by Regime's own clean_up=True).
        """
        node = DummyKnowledgeBaseProducer()
        self_organize = SelfOrganize(algorithms={"kb": node}, device=AVAILABLE_DEVICE)
        targets = [target for _, target, _ in self_organize.get_regime_edges()]
        self.assertNotIn(KnowledgeBase.create, targets)

    def test_wires_in_knowledge_base_create_when_nothing_else_produces_it(
        self,
    ) -> None:
        """
        Without a "knowledge_base"-producing algorithm, get_regime_edges()
        wires the real KnowledgeBase.create to receive "linguistic_variables"
        and "rules" resources directly.
        """

        class Unrelated(Node):
            """A Node whose resource_name has nothing to do with knowledge_base."""

            def __init__(self, resource_name: str = "something_else"):
                super().__init__(resource_name)

            def __call__(self, device: torch.device):
                return None

        self_organize = SelfOrganize(
            algorithms={"other": Unrelated()}, device=AVAILABLE_DEVICE
        )
        edges = self_organize.get_regime_edges()
        self.assertIn(("linguistic_variables", KnowledgeBase.create, 0), edges)
        self.assertIn(("rules", KnowledgeBase.create, 1), edges)


class TestSelfOrganizeRun(unittest.TestCase):
    """
    Covers the full setup() -> run() flow end to end, using a dummy
    "knowledge_base"-producing Node in place of a real CLIP/ECM/WangMendel
    pipeline (see this module's own docstring for why the real pipeline
    cannot currently be exercised here).
    """

    def test_run_returns_the_dummy_nodes_output(self) -> None:
        """run() returns the dummy Node's own computed output (2.0 * 3.0)."""
        node = DummyKnowledgeBaseProducer()
        self_organize = SelfOrganize(
            algorithms={"kb": node}, device=AVAILABLE_DEVICE
        ).setup(
            resources={Resource(name="seed_value", value=2.0)},
            # required_hyperparameters is nested by each dotted component of
            # the Node's __module__ (see regime.utils.module_path_to_dict) -
            # a single key with literal dots in it (e.g. "tests.test_organize")
            # would NOT match.
            configuration={
                "tests": {
                    "test_organize": {"DummyKnowledgeBaseProducer": {"scale": 3.0}}
                }
            },
        )
        self.assertEqual(self_organize.run(), 6.0)

    def test_run_returns_result_of_the_real_knowledge_base_create(self) -> None:
        """
        Without a "knowledge_base"-producing algorithm, run() finds the
        result under KnowledgeBase.create's own module.name key (the
        default branch, as opposed to a Node like FTARM producing it under
        the literal "knowledge_base" key - see the other run() test above).
        An empty LinguisticVariables/rules pair is a valid (if degenerate,
        warning-producing) KnowledgeBase.create() call - confirmed directly.
        """

        class Unrelated(Node):
            """A Node whose resource_name has nothing to do with knowledge_base."""

            def __init__(self, resource_name: str = "something_else"):
                super().__init__(resource_name)

            def __call__(self, device: torch.device):
                # must be non-None: Regime cannot distinguish "this process
                # returned None" from "this resource was never produced" -
                # returning None here would make the *next* vertex look
                # unproduced and raise for an unrelated reason.
                return "irrelevant"

        self_organize = SelfOrganize(
            algorithms={"other": Unrelated()}, device=AVAILABLE_DEVICE
        ).setup(
            resources={
                Resource(
                    name="linguistic_variables",
                    value=LinguisticVariables(inputs=[], targets=[]),
                ),
                Resource(name="rules", value=()),
            },
            configuration={},
        )
        with self.assertWarns(UserWarning):
            result = self_organize.run()
        self.assertIsInstance(result, KnowledgeBase)

    def test_run_raises_when_setup_is_incomplete(self) -> None:
        """
        A Node with some other resource_name still leaves the real
        KnowledgeBase.create wired in (get_regime_edges() only skips it for
        a "knowledge_base"-named producer) - without linguistic_variables/
        rules resources to satisfy it, Regime.start() itself raises
        (a resource it needs was never produced) before run() ever gets a
        results dict to inspect. This is Regime's own error, not
        SelfOrganize.run()'s "KnowledgeBase not found in results" raise -
        see the mocked test below for that one specifically.
        """

        class WrongResourceName(Node):
            """Produces a result under a resource name run() never checks for."""

            def __init__(self, resource_name: str = "not_knowledge_base"):
                super().__init__(resource_name)

            def __call__(self, device: torch.device):
                return "irrelevant"

        self_organize = SelfOrganize(
            algorithms={"wrong": WrongResourceName()}, device=AVAILABLE_DEVICE
        ).setup(resources=set(), configuration={})
        with self.assertRaises(ValueError):
            self_organize.run()

    def test_run_raises_when_neither_access_key_is_present(self) -> None:
        """
        run()'s own final check: even if Regime.start() completes and
        returns a real results dict, run() must still raise if neither
        KnowledgeBase.create's own module.name key nor a literal
        "knowledge_base" key is in it, rather than silently returning
        something else. Under SelfOrganize's normal wiring this branch is
        not reachable through a real run (get_regime_edges() always ends up
        producing one or the other key when a run completes at all - see the
        test above), so Regime.start() is mocked here to isolate run()'s own
        logic directly.
        """
        self_organize = SelfOrganize(
            algorithms={"kb": DummyKnowledgeBaseProducer()}, device=AVAILABLE_DEVICE
        )
        with unittest.mock.patch.object(
            self_organize.regime, "start", return_value={"unrelated_key": "value"}
        ):
            with self.assertRaisesRegex(ValueError, "KnowledgeBase not found"):
                self_organize.run()


if __name__ == "__main__":
    unittest.main()
