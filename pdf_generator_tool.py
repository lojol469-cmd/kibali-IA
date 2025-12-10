#!/usr/bin/env python3
"""
Outil de génération de PDF massif (20-500 pages) avec système de relais IA
Intégré directement dans Kibali comme outil dynamique
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import io


class MassivePDFGenerator:
    """
    Générateur de PDF massif avec système de relais IA
    - Génération par sections (Introduction, Chapitres, Conclusion)
    - Relais automatique entre IAs quand limite de tokens atteinte
    - Formatage professionnel avec ReportLab
    - Support 20-500 pages
    """
    
    def __init__(self, client, model_name="qwen2.5:14b"):
        """
        Args:
            client: Client Ollama pour génération de texte
            model_name: Modèle à utiliser (qwen2.5:14b par défaut)
        """
        self.client = client
        self.model_name = model_name
        self.max_tokens_per_section = 3000  # ~4-6 pages par section
        
    def generate_section(self, prompt, context="", temperature=0.7):
        """
        Génère une section de texte avec l'IA
        
        Args:
            prompt: Instruction pour la section
            context: Contexte des sections précédentes
            temperature: Créativité de la génération
            
        Returns:
            str: Texte généré
        """
        full_prompt = f"""{context}

{prompt}

INSTRUCTIONS:
- Rédige un texte structuré et détaillé
- Minimum 3000 mots (environ 5-6 pages)
- Utilise des paragraphes bien séparés
- Inclus des exemples concrets
- Style académique et professionnel
"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=self.max_tokens_per_section,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_document_structure(self, topic, num_pages):
        """
        Génère la structure du document (chapitres)
        
        Args:
            topic: Sujet du document
            num_pages: Nombre de pages cible
            
        Returns:
            list: Liste des chapitres avec titres et sous-sections
        """
        # Estimer nombre de chapitres (environ 8-10 pages par chapitre)
        num_chapters = max(3, num_pages // 10)
        
        structure_prompt = f"""Crée la structure d'un document de {num_pages} pages sur "{topic}".

Génère une table des matières avec:
- {num_chapters} chapitres principaux
- 3-5 sous-sections par chapitre
- Titres précis et pertinents

Format:
Chapitre 1: [Titre]
  1.1 [Sous-section]
  1.2 [Sous-section]
  ...
Chapitre 2: [Titre]
  2.1 [Sous-section]
  ...
"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": structure_prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        
        structure_text = response.choices[0].message.content
        
        # Parser la structure
        chapters = []
        current_chapter = None
        
        for line in structure_text.split('\n'):
            line = line.strip()
            if line.startswith('Chapitre') or line.startswith('Chapter'):
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = {
                    'title': line,
                    'subsections': []
                }
            elif line and current_chapter and (line[0].isdigit() or line.startswith('-')):
                current_chapter['subsections'].append(line)
        
        if current_chapter:
            chapters.append(current_chapter)
        
        return chapters
    
    def create_pdf(self, topic, num_pages, output_path=None, progress_callback=None):
        """
        Génère le PDF complet
        
        Args:
            topic: Sujet du document
            num_pages: Nombre de pages cible
            output_path: Chemin de sortie (optionnel, sinon BytesIO)
            progress_callback: Fonction callback(step, total, message)
            
        Returns:
            bytes: Contenu du PDF
        """
        # Créer le buffer
        if output_path:
            buffer = open(output_path, 'wb')
        else:
            buffer = io.BytesIO()
        
        # Créer le document PDF
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Style pour le titre principal
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Style pour les chapitres
        chapter_style = ParagraphStyle(
            'ChapterTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        # Style pour les sous-sections
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        # Style pour le corps de texte
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=16
        )
        
        # Liste des éléments du PDF
        story = []
        
        # PAGE DE TITRE
        if progress_callback:
            progress_callback(0, 100, "Génération de la page de titre...")
        
        story.append(Spacer(1, 4*cm))
        story.append(Paragraph(topic.upper(), title_style))
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(f"Document de {num_pages} pages", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Par Kibali IA", styles['Normal']))
        story.append(PageBreak())
        
        # GÉNÉRATION DE LA STRUCTURE
        if progress_callback:
            progress_callback(5, 100, "Création de la structure du document...")
        
        chapters = self.generate_document_structure(topic, num_pages)
        
        # TABLE DES MATIÈRES
        story.append(Paragraph("TABLE DES MATIÈRES", chapter_style))
        story.append(Spacer(1, 0.5*cm))
        
        for i, chapter in enumerate(chapters, 1):
            story.append(Paragraph(f"{chapter['title']}", body_style))
            for subsection in chapter['subsections']:
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{subsection}", body_style))
        
        story.append(PageBreak())
        
        # INTRODUCTION
        if progress_callback:
            progress_callback(10, 100, "Génération de l'introduction...")
        
        story.append(Paragraph("INTRODUCTION", chapter_style))
        story.append(Spacer(1, 0.5*cm))
        
        intro_prompt = f"""Rédige une introduction détaillée pour un document sur "{topic}".

CONTENU DE L'INTRODUCTION:
- Contexte général et importance du sujet
- Problématiques abordées
- Objectifs du document
- Plan et structure

Longueur: 2-3 pages (environ 1500 mots)
Style: Académique et engageant
"""
        
        intro_text = self.generate_section(intro_prompt, temperature=0.7)
        
        # Découper en paragraphes
        for para in intro_text.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
                story.append(Spacer(1, 0.3*cm))
        
        story.append(PageBreak())
        
        # CHAPITRES
        context = f"Document sur: {topic}\n\nIntroduction: {intro_text[:500]}...\n\n"
        
        total_chapters = len(chapters)
        for chapter_idx, chapter in enumerate(chapters, 1):
            progress = 10 + int((chapter_idx / total_chapters) * 80)
            
            if progress_callback:
                progress_callback(progress, 100, f"Génération {chapter['title']}...")
            
            # Titre du chapitre
            story.append(Paragraph(chapter['title'], chapter_style))
            story.append(Spacer(1, 0.5*cm))
            
            # Générer le contenu du chapitre
            chapter_prompt = f"""Rédige le contenu complet pour: {chapter['title']}

SOUS-SECTIONS À COUVRIR:
{chr(10).join(chapter['subsections'])}

INSTRUCTIONS:
- Développe chaque sous-section en profondeur
- Utilise des exemples concrets et des cas pratiques
- Inclus des définitions claires
- Ajoute des analyses détaillées
- Longueur: 8-10 pages (environ 5000 mots)
- Style: Précis, structuré et académique
"""
            
            chapter_text = self.generate_section(chapter_prompt, context, temperature=0.7)
            
            # Ajouter au contexte pour la continuité
            context += f"\n\n{chapter['title']}: {chapter_text[:300]}...\n"
            
            # Découper et formater
            for para in chapter_text.split('\n\n'):
                if para.strip():
                    # Détecter les sous-titres
                    if any(para.strip().startswith(prefix) for prefix in ['#', '##', '1.', '2.', '3.', 'A.', 'B.']):
                        story.append(Paragraph(para.strip(), section_style))
                    else:
                        story.append(Paragraph(para.strip(), body_style))
                    story.append(Spacer(1, 0.3*cm))
            
            story.append(PageBreak())
        
        # CONCLUSION
        if progress_callback:
            progress_callback(90, 100, "Génération de la conclusion...")
        
        story.append(Paragraph("CONCLUSION", chapter_style))
        story.append(Spacer(1, 0.5*cm))
        
        conclusion_prompt = f"""Rédige une conclusion complète pour le document sur "{topic}".

ÉLÉMENTS À INCLURE:
- Synthèse des points clés abordés
- Réponses aux problématiques initiales
- Perspectives et ouvertures
- Recommandations pratiques

Longueur: 2-3 pages (environ 1500 mots)
Style: Synthétique mais complet
"""
        
        conclusion_text = self.generate_section(conclusion_prompt, context, temperature=0.7)
        
        for para in conclusion_text.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
                story.append(Spacer(1, 0.3*cm))
        
        # CONSTRUIRE LE PDF
        if progress_callback:
            progress_callback(95, 100, "Construction du PDF...")
        
        doc.build(story)
        
        # Récupérer les bytes
        if output_path:
            buffer.close()
            with open(output_path, 'rb') as f:
                pdf_bytes = f.read()
        else:
            pdf_bytes = buffer.getvalue()
            buffer.close()
        
        if progress_callback:
            progress_callback(100, 100, "PDF généré avec succès!")
        
        return pdf_bytes


def generate_massive_pdf(topic, num_pages, client, model_name="qwen2.5:14b", progress_callback=None):
    """
    Fonction helper pour générer un PDF massif
    
    Args:
        topic: Sujet du document
        num_pages: Nombre de pages (20-500)
        client: Client Ollama
        model_name: Modèle IA à utiliser
        progress_callback: Fonction de progression
        
    Returns:
        bytes: Contenu du PDF
    """
    generator = MassivePDFGenerator(client, model_name)
    return generator.create_pdf(topic, num_pages, progress_callback=progress_callback)
