import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Plugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class TranslationPlugin(Plugin):
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str, context: dict = None) -> str:
        pass


class OutputPlugin(Plugin):
    @abstractmethod
    def write(self, segments: list, output_path: Path, **kwargs) -> bool:
        pass


class PluginManager:
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugins: dict[str, Plugin] = {}
        self.plugin_dir = plugin_dir or Path("plugins")

    def discover_plugins(self) -> list[str]:
        discovered = []

        if not self.plugin_dir.exists():
            logger.info(f"Plugin directory does not exist: {self.plugin_dir}")
            return discovered

        for file in self.plugin_dir.glob("*.py"):
            if file.stem.startswith("_"):
                continue
            discovered.append(file.stem)

        return discovered

    def load_plugin(self, plugin_name: str) -> bool:
        try:
            plugin_path = self.plugin_dir / f"{plugin_name}.py"
            if not plugin_path.exists():
                logger.error(f"Plugin file not found: {plugin_path}")
                return False

            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "register"):
                    plugin = module.register()
                    self.plugins[plugin.name] = plugin
                    logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                    return True
                logger.warning(f"Plugin {plugin_name} has no register() function")
                return False

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def load_all(self) -> int:
        count = 0
        for plugin_name in self.discover_plugins():
            if self.load_plugin(plugin_name):
                count += 1
        return count

    def get_plugin(self, name: str) -> Optional[Plugin]:
        return self.plugins.get(name)

    def list_plugins(self) -> list[dict[str, str]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description
            }
            for p in self.plugins.values()
        ]

    def unload_plugin(self, name: str) -> bool:
        if name in self.plugins:
            self.plugins[name].cleanup()
            del self.plugins[name]
            return True
        return False


_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
