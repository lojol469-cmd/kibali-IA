"""
Module Outils - Système d'outils dynamiques pour Kibali
Permet de charger et utiliser des outils (IA, logiciels) de manière dynamique
selon le contexte des questions utilisateur.
"""

import os
import importlib
import inspect
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Classe de base pour tous les outils dynamiques"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de l'outil"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description de l'outil"""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Liste des capacités/compétences de l'outil"""
        pass

    @abstractmethod
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """
        Détermine si l'outil peut traiter cette requête
        Retourne un score de confiance (0.0 à 1.0)
        """
        pass

    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Exécute l'outil avec la requête donnée
        Retourne un dictionnaire avec les résultats
        """
        pass

class ToolManager:
    """Gestionnaire d'outils dynamiques"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._load_tools()

    def _load_tools(self):
        """Charge tous les outils disponibles dans le dossier"""
        tools_dir = os.path.dirname(__file__)

        for filename in os.listdir(tools_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]  # Remove .py extension
                try:
                    # Import the module
                    module = importlib.import_module(f'outils.{module_name}')

                    # Find all classes that inherit from BaseTool
                    # Comparer par nom plutôt que par instance pour éviter problèmes d'import
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            hasattr(obj, '__bases__') and
                            any('BaseTool' in str(base) for base in obj.__bases__) and
                            'BaseTool' not in obj.__name__):
                            # Instantiate the tool
                            tool_instance = obj()
                            self.tools[tool_instance.name] = tool_instance
                            print(f"✅ Outil chargé: {tool_instance.name}")

                except Exception as e:
                    print(f"❌ Erreur chargement outil {module_name}: {e}")

    def get_relevant_tools(self, query: str, context: Dict[str, Any] = None, max_tools: int = 3) -> List[BaseTool]:
        """
        Retourne les outils les plus pertinents pour une requête
        """
        tool_scores = []
        for tool in self.tools.values():
            score = tool.can_handle(query, context)
            if score > 0.0:
                tool_scores.append((tool, score))

        # Trier par score décroissant
        tool_scores.sort(key=lambda x: x[1], reverse=True)

        # Retourner les meilleurs outils
        return [tool for tool, score in tool_scores[:max_tools]]

    def execute_tools(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Exécute les outils pertinents et retourne leurs résultats
        """
        relevant_tools = self.get_relevant_tools(query, context)
        results = []

        for tool in relevant_tools:
            try:
                print(f"🔧 Exécution outil: {tool.name}")
                result = tool.execute(query, context)
                results.append({
                    'tool_name': tool.name,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                print(f"❌ Erreur exécution outil {tool.name}: {e}")
                results.append({
                    'tool_name': tool.name,
                    'error': str(e),
                    'success': False
                })

        return results

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Retourne la liste de tous les outils disponibles"""
        return {
            name: {
                'description': tool.description,
                'capabilities': tool.capabilities
            }
            for name, tool in self.tools.items()
        }

# Instance globale du gestionnaire d'outils
tool_manager = ToolManager()

# Exports directs pour compatibilité
from .pdf_generator_tool import generate_massive_pdf, MassivePDFGenerator

try:
    from .excel_ai_organizer import ExcelAIOrganizer, excel_organizer_tool
    EXCEL_AVAILABLE = True
    # Note: organize_excel_data n'est pas une classe mais une fonction dans le fichier
except ImportError as e:
    print(f"⚠️ Excel AI Organizer non disponible: {e}")
    ExcelAIOrganizer = None
    excel_organizer_tool = None
    EXCEL_AVAILABLE = False

__all__ = [
    'BaseTool',
    'ToolManager', 
    'tool_manager',
    'generate_massive_pdf',
    'MassivePDFGenerator',
    'ExcelAIOrganizer',
    'excel_organizer_tool',
    'EXCEL_AVAILABLE'
]