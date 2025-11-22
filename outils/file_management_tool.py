"""
Outil de Gestion de Fichiers
Gère les opérations sur les fichiers et dossiers
"""

import os
import shutil
import re
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool

class FileManagementTool(BaseTool):
    """Outil spécialisé dans la gestion de fichiers et dossiers"""

    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return "Gestionnaire de fichiers avec opérations CRUD complètes"

    @property
    def capabilities(self) -> List[str]:
        return [
            "création de fichiers",
            "lecture de fichiers",
            "écriture dans des fichiers",
            "suppression de fichiers",
            "copie de fichiers",
            "déplacement de fichiers",
            "création de dossiers",
            "listage de répertoires",
            "recherche de fichiers",
            "analyse de fichiers",
            "compression/décompression",
            "synchronisation de fichiers"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête concerne la gestion de fichiers"""
        file_keywords = [
            "fichier", "file", "dossier", "folder", "répertoire", "directory",
            "créer", "create", "lire", "read", "écrire", "write",
            "supprimer", "delete", "remove", "copier", "copy",
            "déplacer", "move", "renommer", "rename", "chercher", "search",
            "trouver", "find", "lister", "list", "analyser", "analyze",
            "compresser", "zip", "extraire", "unzip"
        ]

        file_extensions = [
            ".txt", ".py", ".js", ".html", ".css", ".json", ".xml",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".png", ".gif"
        ]

        query_lower = query.lower()

        # Mots-clés de gestion de fichiers
        score = 0.0
        for keyword in file_keywords:
            if keyword in query_lower:
                score += 0.3

        # Extensions de fichiers mentionnées
        for ext in file_extensions:
            if ext in query_lower:
                score += 0.4

        # Chemins de fichiers détectés
        path_patterns = [
            r'/[\w/.-]+',  # Chemins Unix
            r'C:\\[\w\\.-]+',  # Chemins Windows
            r'\w+\.\w+'  # Noms de fichiers avec extension
        ]

        for pattern in path_patterns:
            if re.search(pattern, query):
                score += 0.5

        # Contexte avec chemin de fichier
        if context and any(key in context for key in ['file_path', 'directory_path', 'filename']):
            score += 0.6

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Exécute l'opération de gestion de fichiers demandée"""
        try:
            # Identifier le type d'opération
            operation_type = self._identify_operation_type(query)

            if operation_type == "create_file":
                result = self._create_file_operation(query, context)
            elif operation_type == "read_file":
                result = self._read_file_operation(query, context)
            elif operation_type == "write_file":
                result = self._write_file_operation(query, context)
            elif operation_type == "delete_file":
                result = self._delete_file_operation(query, context)
            elif operation_type == "copy_file":
                result = self._copy_file_operation(query, context)
            elif operation_type == "move_file":
                result = self._move_file_operation(query, context)
            elif operation_type == "create_directory":
                result = self._create_directory_operation(query, context)
            elif operation_type == "list_directory":
                result = self._list_directory_operation(query, context)
            elif operation_type == "search_files":
                result = self._search_files_operation(query, context)
            elif operation_type == "analyze_file":
                result = self._analyze_file_operation(query, context)
            else:
                result = {'error': 'Type d\'opération non reconnu'}

            return {
                'query': query,
                'operation_type': operation_type,
                'result': result,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de l'opération fichier: {str(e)}",
                'query': query
            }

    def _identify_operation_type(self, query: str) -> str:
        """Identifie le type d'opération demandée"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["créer", "create", "nouveau"]):
            if any(word in query_lower for word in ["fichier", "file"]):
                return "create_file"
            elif any(word in query_lower for word in ["dossier", "directory", "folder"]):
                return "create_directory"

        elif any(word in query_lower for word in ["lire", "read", "ouvrir", "open"]):
            return "read_file"

        elif any(word in query_lower for word in ["écrire", "write", "modifier", "edit"]):
            return "write_file"

        elif any(word in query_lower for word in ["supprimer", "delete", "remove", "effacer"]):
            return "delete_file"

        elif any(word in query_lower for word in ["copier", "copy"]):
            return "copy_file"

        elif any(word in query_lower for word in ["déplacer", "move", "renommer", "rename"]):
            return "move_file"

        elif any(word in query_lower for word in ["lister", "list", "contenu"]):
            return "list_directory"

        elif any(word in query_lower for word in ["chercher", "search", "find", "trouver"]):
            return "search_files"

        elif any(word in query_lower for word in ["analyser", "analyze", "info", "information"]):
            return "analyze_file"

        return "unknown"

    def _extract_file_path(self, query: str, context: Dict[str, Any] = None) -> str:
        """Extrait le chemin de fichier de la requête"""
        # Vérifier le contexte d'abord
        if context:
            for key in ['file_path', 'filepath', 'path']:
                if key in context:
                    return context[key]

        # Chercher des chemins dans la requête
        path_patterns = [
            r'["\']([^"\']+\.\w+)["\']',  # Chemins entre guillemets
            r'/[\w/.-]+\.\w+',  # Chemins Unix absolus
            r'\w+\.\w+',  # Noms de fichiers simples
            r'C:\\[\w\\.-]+'  # Chemins Windows
        ]

        for pattern in path_patterns:
            match = re.search(pattern, query)
            if match:
                path = match.group(1) if match.groups() else match.group(0)
                # Nettoyer le chemin
                path = path.strip('"\'')

                # Rendre le chemin absolu si nécessaire
                if not os.path.isabs(path):
                    # Utiliser le répertoire de travail actuel
                    path = os.path.join(os.getcwd(), path)

                return path

        return ""

    def _create_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crée un nouveau fichier"""
        filename = self._extract_file_path(query, context)

        if not filename:
            # Demander le nom du fichier
            filename = "nouveau_fichier.txt"

        try:
            # Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Créer le fichier
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Fichier créé automatiquement\n")

            return {
                'action': 'file_created',
                'filename': filename,
                'size': 0,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'file_creation_failed',
                'filename': filename,
                'error': str(e),
                'success': False
            }

    def _read_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Lit le contenu d'un fichier"""
        filename = self._extract_file_path(query, context)

        if not filename or not os.path.exists(filename):
            return {
                'action': 'file_read_failed',
                'error': 'Fichier non trouvé',
                'filename': filename
            }

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                'action': 'file_read',
                'filename': filename,
                'content': content,
                'size': len(content),
                'lines': len(content.split('\n')),
                'success': True
            }

        except Exception as e:
            return {
                'action': 'file_read_failed',
                'filename': filename,
                'error': str(e),
                'success': False
            }

    def _write_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Écrit dans un fichier"""
        filename = self._extract_file_path(query, context)

        # Extraire le contenu à écrire
        content_match = re.search(r'(?:contenu|content|texte|text)\s*:\s*["\']([^"\']+)["\']', query, re.IGNORECASE)
        if content_match:
            content = content_match.group(1)
        elif context and 'content' in context:
            content = context['content']
        else:
            content = "Contenu ajouté automatiquement"

        if not filename:
            return {'error': 'Nom de fichier non spécifié'}

        try:
            # Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Écrire dans le fichier
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                'action': 'file_written',
                'filename': filename,
                'content_length': len(content),
                'success': True
            }

        except Exception as e:
            return {
                'action': 'file_write_failed',
                'filename': filename,
                'error': str(e),
                'success': False
            }

    def _delete_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Supprime un fichier"""
        filename = self._extract_file_path(query, context)

        if not filename or not os.path.exists(filename):
            return {
                'action': 'file_delete_failed',
                'error': 'Fichier non trouvé',
                'filename': filename
            }

        try:
            # Vérifier si c'est un dossier
            if os.path.isdir(filename):
                shutil.rmtree(filename)
                action = 'directory_deleted'
            else:
                os.remove(filename)
                action = 'file_deleted'

            return {
                'action': action,
                'filename': filename,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'delete_failed',
                'filename': filename,
                'error': str(e),
                'success': False
            }

    def _copy_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Copie un fichier"""
        # Extraire les chemins source et destination
        paths = re.findall(r'["\']([^"\']+)["\']', query)
        if len(paths) >= 2:
            source_path = paths[0]
            dest_path = paths[1]
        else:
            return {'error': 'Chemins source et destination non spécifiés'}

        try:
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)

            return {
                'action': 'file_copied',
                'source': source_path,
                'destination': dest_path,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'copy_failed',
                'source': source_path,
                'destination': dest_path,
                'error': str(e),
                'success': False
            }

    def _move_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Déplace ou renomme un fichier"""
        paths = re.findall(r'["\']([^"\']+)["\']', query)
        if len(paths) >= 2:
            source_path = paths[0]
            dest_path = paths[1]
        else:
            return {'error': 'Chemins source et destination non spécifiés'}

        try:
            shutil.move(source_path, dest_path)

            return {
                'action': 'file_moved',
                'source': source_path,
                'destination': dest_path,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'move_failed',
                'source': source_path,
                'destination': dest_path,
                'error': str(e),
                'success': False
            }

    def _create_directory_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crée un dossier"""
        # Extraire le nom du dossier
        dir_match = re.search(r'(?:dossier|directory|folder)\s+["\']([^"\']+)["\']', query, re.IGNORECASE)
        if dir_match:
            dirname = dir_match.group(1)
        elif context and 'directory_path' in context:
            dirname = context['directory_path']
        else:
            dirname = "nouveau_dossier"

        try:
            os.makedirs(dirname, exist_ok=True)

            return {
                'action': 'directory_created',
                'directory': dirname,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'directory_creation_failed',
                'directory': dirname,
                'error': str(e),
                'success': False
            }

    def _list_directory_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Liste le contenu d'un dossier"""
        # Extraire le chemin du dossier
        dir_match = re.search(r'(?:dossier|directory|folder)\s+["\']([^"\']+)["\']', query, re.IGNORECASE)
        if dir_match:
            dirname = dir_match.group(1)
        elif context and 'directory_path' in context:
            dirname = context['directory_path']
        else:
            dirname = "."

        try:
            if not os.path.exists(dirname):
                return {
                    'action': 'list_failed',
                    'error': 'Dossier non trouvé',
                    'directory': dirname
                }

            items = os.listdir(dirname)
            files = []
            directories = []

            for item in items:
                full_path = os.path.join(dirname, item)
                if os.path.isdir(full_path):
                    directories.append({
                        'name': item,
                        'type': 'directory',
                        'path': full_path
                    })
                else:
                    stat = os.stat(full_path)
                    files.append({
                        'name': item,
                        'type': 'file',
                        'path': full_path,
                        'size': stat.st_size,
                        'modified': stat.st_mtime
                    })

            return {
                'action': 'directory_listed',
                'directory': dirname,
                'files': files,
                'directories': directories,
                'total_items': len(items),
                'success': True
            }

        except Exception as e:
            return {
                'action': 'list_failed',
                'directory': dirname,
                'error': str(e),
                'success': False
            }

    def _search_files_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recherche des fichiers"""
        # Extraire le terme de recherche
        search_match = re.search(r'(?:chercher|search|find)\s+["\']([^"\']+)["\']', query, re.IGNORECASE)
        if search_match:
            search_term = search_match.group(1)
        elif context and 'search_term' in context:
            search_term = context['search_term']
        else:
            return {'error': 'Terme de recherche non spécifié'}

        # Extraire le dossier de recherche
        dir_match = re.search(r'(?:dans|in)\s+["\']([^"\']+)["\']', query, re.IGNORECASE)
        search_dir = dir_match.group(1) if dir_match else "."

        try:
            found_files = []

            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if search_term.lower() in file.lower():
                        full_path = os.path.join(root, file)
                        stat = os.stat(full_path)
                        found_files.append({
                            'name': file,
                            'path': full_path,
                            'size': stat.st_size,
                            'modified': stat.st_mtime
                        })

            return {
                'action': 'files_searched',
                'search_term': search_term,
                'search_directory': search_dir,
                'found_files': found_files,
                'total_found': len(found_files),
                'success': True
            }

        except Exception as e:
            return {
                'action': 'search_failed',
                'search_term': search_term,
                'search_directory': search_dir,
                'error': str(e),
                'success': False
            }

    def _analyze_file_operation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyse un fichier"""
        filename = self._extract_file_path(query, context)

        if not filename or not os.path.exists(filename):
            return {
                'action': 'analyze_failed',
                'error': 'Fichier non trouvé',
                'filename': filename
            }

        try:
            stat = os.stat(filename)
            analysis = {
                'filename': filename,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'created': stat.st_ctime,
                'is_directory': os.path.isdir(filename),
                'extension': os.path.splitext(filename)[1],
                'readable': os.access(filename, os.R_OK),
                'writable': os.access(filename, os.W_OK),
                'executable': os.access(filename, os.X_OK)
            }

            # Analyse spécifique selon le type de fichier
            if os.path.isfile(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read(1024)  # Premier kilo-octet
                        analysis['preview'] = content
                        analysis['line_count'] = len(content.split('\n'))
                        analysis['word_count'] = len(content.split())
                        analysis['char_count'] = len(content)
                except:
                    analysis['binary_file'] = True

            return {
                'action': 'file_analyzed',
                'filename': filename,
                'analysis': analysis,
                'success': True
            }

        except Exception as e:
            return {
                'action': 'analyze_failed',
                'filename': filename,
                'error': str(e),
                'success': False
            }