import ast
import unittest
from pathlib import Path


class VisualizationContractTest(unittest.TestCase):
    def test_dataset_modules_do_not_define_visualization_paths(self) -> None:
        dataset_files = sorted(Path("src/depth_collector/datasets").glob("*.py"))
        forbidden_function_names = {
            "visualize",
            "render_visualization",
            "create_contact_sheet",
            "load_processed_samples",
        }

        for path in dataset_files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "depth_collector.visualization":
                    self.fail(f"{path} must not import from depth_collector.visualization")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "depth_collector.visualization":
                            self.fail(f"{path} must not import depth_collector.visualization")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_function_names:
                    self.fail(f"{path} must not define visualization helper {node.name}")


if __name__ == "__main__":
    unittest.main()
