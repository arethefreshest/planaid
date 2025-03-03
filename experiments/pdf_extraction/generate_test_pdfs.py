"""
Test PDF Generator

This script generates test PDFs with various layouts and content types to evaluate
extraction performance. It creates:
1. Simple single-column text
2. Multi-column text
3. Mixed content (text + tables)
4. Scanned-like content (rendered as images)
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from pathlib import Path
import lorem
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import io

class TestPDFGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Add custom style for columns
        self.styles.add(ParagraphStyle(
            name='TwoColumns',
            parent=self.styles['Normal'],
            spaceBefore=12,
            spaceAfter=12,
        ))

    def generate_single_column(self):
        """Generate a simple single-column PDF."""
        doc = SimpleDocTemplate(
            self.output_dir / "single_column.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Add title
        story.append(Paragraph("Single Column Test Document", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add paragraphs
        for _ in range(5):
            story.append(Paragraph(lorem.paragraph(), self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        doc.build(story)
        return "single_column.pdf"

    def generate_multi_column(self):
        """Generate a PDF with multiple columns."""
        from reportlab.platypus import Frame, PageTemplate
        
        doc = SimpleDocTemplate(
            self.output_dir / "multi_column.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create two columns
        frame1 = Frame(doc.leftMargin, doc.bottomMargin,
                      (doc.width-doc.rightMargin)/2-6, doc.height,
                      id='col1')
        frame2 = Frame(doc.leftMargin + (doc.width-doc.rightMargin)/2+6,
                      doc.bottomMargin, (doc.width-doc.rightMargin)/2-6,
                      doc.height, id='col2')
        
        doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2])])
        
        story = []
        story.append(Paragraph("Multi-Column Test Document", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add content to columns
        for _ in range(8):
            story.append(Paragraph(lorem.paragraph(), self.styles['TwoColumns']))
            story.append(Spacer(1, 12))
        
        doc.build(story)
        return "multi_column.pdf"

    def generate_mixed_content(self):
        """Generate a PDF with mixed content (text, tables, etc.)."""
        doc = SimpleDocTemplate(
            self.output_dir / "mixed_content.pdf",
            pagesize=letter
        )
        
        story = []
        story.append(Paragraph("Mixed Content Test Document", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add some text
        story.append(Paragraph(lorem.paragraph(), self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add a table
        data = [['Header 1', 'Header 2', 'Header 3'],
                ['Row 1', '123', 'ABC'],
                ['Row 2', '456', 'DEF'],
                ['Row 3', '789', 'GHI']]
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(Spacer(1, 12))
        
        # Add more text
        story.append(Paragraph(lorem.paragraph(), self.styles['Normal']))
        
        doc.build(story)
        return "mixed_content.pdf"

    def generate_scanned_like(self):
        """Generate a PDF that looks like a scanned document."""
        # Create an image with text
        img = PILImage.new('RGB', (2000, 2800), color='white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        except:
            font = ImageFont.load_default()
        
        # Add text to image
        text = "Scanned-like Test Document\n\n" + "\n\n".join([lorem.paragraph() for _ in range(3)])
        d.text((100, 100), text, font=font, fill='black')
        
        # Add some noise to make it look scanned
        noise = np.random.normal(0, 5, img.size[::-1]).astype(np.uint8)
        noisy_img = PILImage.fromarray(np.array(img) + noise)
        
        # Convert to PDF
        pdf_path = self.output_dir / "scanned_like.pdf"
        noisy_img.save(str(pdf_path), "PDF", resolution=100.0)
        return "scanned_like.pdf"

    def generate_all(self):
        """Generate all test PDFs."""
        files = []
        files.append(self.generate_single_column())
        files.append(self.generate_multi_column())
        files.append(self.generate_mixed_content())
        files.append(self.generate_scanned_like())
        
        # Create ground truth text files
        for pdf_file in files:
            base_name = Path(pdf_file).stem
            with open(self.output_dir / f"{base_name}_truth.txt", 'w') as f:
                # For now, just write some placeholder text
                f.write(f"Ground truth for {base_name}\n")
                f.write(lorem.paragraph())
        
        return files

def main():
    generator = TestPDFGenerator("test_pdfs")
    files = generator.generate_all()
    print(f"Generated test PDFs: {files}")

if __name__ == "__main__":
    main() 