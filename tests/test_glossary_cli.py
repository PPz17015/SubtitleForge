"""
Tests for Glossary CLI command.

Coverage:
- Path type conversion (click.Path returns str, not Path)
- Add/remove/list/export/import actions
- File persistence (save + load roundtrip)
- CSV export/import roundtrip
- Error handling for missing arguments
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from click.testing import CliRunner

from main import cli


class TestGlossaryAdd:
    """Test the 'glossary add' action."""

    def test_add_term_without_file(self):
        """Test adding a term without saving to file."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'glossary', 'add',
            '--source', 'Hello',
            '--target', 'Xin chào'
        ])

        assert result.exit_code == 0
        assert "Added" in result.output
        assert "Hello" in result.output

    def test_add_term_with_file(self):
        """Test adding a term and saving to glossary file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            # Write valid empty glossary JSON (load_from_file needs valid JSON)
            json.dump({"version": "1.0", "terms": {}}, f)
            temp_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli, [
                'glossary', 'add',
                '--glossary', temp_path,
                '--source', 'ありがとう',
                '--target', 'Cảm ơn',
                '--context', 'greeting'
            ])

            assert result.exit_code == 0
            assert "Added and saved" in result.output

            # Verify file contains the term
            with open(temp_path, encoding='utf-8') as f:
                data = json.load(f)

            assert "terms" in data
            # Terms are stored by lowercase key
            assert "ありがとう" in data["terms"]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_add_term_missing_source(self):
        """Test that add without --source shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'glossary', 'add',
            '--target', 'something'
        ])

        assert result.exit_code == 0
        assert "Error" in result.output or "--source" in result.output

    def test_add_term_missing_target(self):
        """Test that add without --target shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'glossary', 'add',
            '--source', 'something'
        ])

        assert result.exit_code == 0
        assert "Error" in result.output or "--target" in result.output


class TestGlossaryRemove:
    """Test the 'glossary remove' action."""

    def test_remove_term(self):
        """Test removing a term from glossary file."""
        # First create a glossary file with a term
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "terms": {
                    "hello": {
                        "source": "Hello",
                        "target": "Xin chào",
                        "context": "greeting",
                        "notes": ""
                    }
                }
            }, f, ensure_ascii=False)
            temp_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli, [
                'glossary', 'remove',
                '--glossary', temp_path,
                '--source', 'Hello'
            ])

            assert result.exit_code == 0
            assert "Removed" in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_remove_missing_source_param(self):
        """Test remove without --source shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'glossary', 'remove'
        ])

        assert result.exit_code == 0
        assert "Error" in result.output or "--source" in result.output


class TestGlossaryList:
    """Test the 'glossary list' action."""

    def test_list_empty(self):
        """Test listing empty glossary."""
        runner = CliRunner()
        result = runner.invoke(cli, ['glossary', 'list'])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_list_with_terms(self):
        """Test listing glossary with existing terms."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "terms": {
                    "hello": {
                        "source": "Hello",
                        "target": "Xin chào",
                        "context": "greeting",
                        "notes": ""
                    },
                    "goodbye": {
                        "source": "Goodbye",
                        "target": "Tạm biệt",
                        "context": "farewell",
                        "notes": ""
                    }
                }
            }, f, ensure_ascii=False)
            temp_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli, [
                'glossary', 'list',
                '--glossary', temp_path
            ])

            assert result.exit_code == 0
            assert "Hello" in result.output
            assert "Goodbye" in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestGlossaryExport:
    """Test the 'glossary export' action."""

    def test_export_requires_glossary_flag(self):
        """Test that export without --glossary shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ['glossary', 'export'])

        assert result.exit_code == 0
        assert "Error" in result.output or "--glossary" in result.output

    def test_export_csv(self):
        """Test exporting glossary to CSV."""
        # Create a glossary JSON first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "terms": {
                    "hello": {
                        "source": "Hello",
                        "target": "Xin chào",
                        "context": "greeting",
                        "notes": "common"
                    }
                }
            }, f, ensure_ascii=False)
            json_path = f.name

        csv_path = json_path.replace('.json', '.csv')

        try:
            # Load glossary, then export to CSV
            from core.batch_processing import GlossaryManager
            manager = GlossaryManager.load_from_file(Path(json_path))
            manager.export_csv(Path(csv_path))

            # Verify CSV content
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["Source", "Target", "Context", "Notes"]  # header
            assert rows[1][0] == "Hello"
            assert rows[1][1] == "Xin chào"
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(csv_path).unlink(missing_ok=True)


class TestGlossaryImport:
    """Test the 'glossary import' action."""

    def test_import_requires_glossary_flag(self):
        """Test that import without --glossary shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ['glossary', 'import'])

        assert result.exit_code == 0
        assert "Error" in result.output or "--glossary" in result.output

    def test_import_csv(self):
        """Test importing glossary from CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                         encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Source", "Target", "Context", "Notes"])
            writer.writerow(["Hello", "Xin chào", "greeting", ""])
            writer.writerow(["Goodbye", "Tạm biệt", "farewell", ""])
            csv_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli, [
                'glossary', 'import',
                '--glossary', csv_path
            ])

            assert result.exit_code == 0
            assert "Imported" in result.output
            assert "2 terms" in result.output
        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestGlossaryRoundtrip:
    """Test glossary persistence roundtrips."""

    def test_json_roundtrip(self):
        """
        Test: add term → save to JSON → load from JSON → verify term exists.

        This catches the Path type bug where click.Path() returns str
        but the function expected a Path object.
        """
        from core.batch_processing import GlossaryManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create and save
            manager = GlossaryManager()
            manager.add_term("先生", "Thầy/Cô", "education")
            manager.add_term("生徒", "Học sinh", "education")
            manager.save_to_file(temp_path)

            # Load and verify
            loaded = GlossaryManager.load_from_file(temp_path)
            assert loaded.has_term("先生")
            assert loaded.has_term("生徒")
            assert loaded.get_translation("先生") == "Thầy/Cô"
            assert loaded.get_translation("生徒") == "Học sinh"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_csv_roundtrip(self):
        """Test: add terms → export CSV → import CSV → verify terms match."""
        from core.batch_processing import GlossaryManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = Path(f.name)

        try:
            # Create and export
            original = GlossaryManager()
            original.add_term("母", "Mẹ", "family")
            original.add_term("息子", "Con trai", "family")
            original.export_csv(csv_path)

            # Import and verify
            imported = GlossaryManager.import_csv(csv_path)
            assert imported.has_term("母")
            assert imported.has_term("息子")
            assert imported.get_translation("母") == "Mẹ"
            assert imported.get_translation("息子") == "Con trai"
        finally:
            csv_path.unlink(missing_ok=True)

    def test_path_type_conversion_via_cli(self):
        """
        REGRESSION TEST: Verify that the glossary CLI command
        correctly handles click.Path() returning a string,
        not a Path object.

        This directly tests the bug fix where `glossary_path.exists()`
        was called on a string (which doesn't have `.exists()` method).
        """
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            # Write a valid empty glossary so load_from_file doesn't fail
            json.dump({"version": "1.0", "terms": {}}, f)
            temp_path = f.name  # This is a string, like click.Path() returns

        try:
            runner = CliRunner()

            # This should NOT crash with AttributeError on str.exists()
            result = runner.invoke(cli, [
                'glossary', 'add',
                '--glossary', temp_path,
                '--source', 'テスト',
                '--target', 'Kiểm tra'
            ])

            assert result.exit_code == 0
            assert "exception" not in result.output.lower()
            assert "Added and saved" in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestGlossaryHelp:
    """Test glossary command help."""

    def test_glossary_help(self):
        """Test glossary command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['glossary', '--help'])

        assert result.exit_code == 0
        assert "--glossary" in result.output or "glossary" in result.output.lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
