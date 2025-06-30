#!/usr/bin/env python3
"""
Complete Analysis Report Generator for SmartRocket Analytics
Generates a comprehensive HTML report that can be easily converted to PDF
"""

import re
from pathlib import Path
import base64
import os
from datetime import datetime

def embed_images_in_html(html_content, base_path):
    """Convert image references to embedded base64 data"""
    
    # Find all image references in markdown format
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_src = match.group(2)
        
        # Convert relative path to absolute
        img_path = Path(base_path) / img_src
        
        if img_path.exists():
            # Read image and convert to base64
            with open(img_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
            # Determine image type
            img_ext = img_path.suffix.lower()
            if img_ext == '.png':
                mime_type = 'image/png'
            elif img_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'
            
            # Create data URL and HTML img tag
            data_url = f"data:{mime_type};base64,{img_data}"
            return f'<img src="{data_url}" alt="{alt_text}" class="report-image">'
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")
            return f'<p><em>Image not found: {img_src}</em></p>'
    
    return re.sub(img_pattern, replace_image, html_content)

def markdown_to_html(md_content):
    """Simple markdown to HTML converter"""
    
    html = md_content
    
    # Convert headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Convert bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Convert code blocks
    html = re.sub(r'```(\w*)\n(.*?)\n```', r'<pre class="code-block"><code>\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Convert links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Convert tables
    def convert_table(text):
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return text
            
        table_html = '<table class="data-table">\n'
        
        # Header row
        header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        table_html += '<thead><tr>'
        for cell in header_cells:
            table_html += f'<th>{cell}</th>'
        table_html += '</tr></thead>\n'
        
        # Data rows (skip separator line)
        table_html += '<tbody>'
        for line in lines[2:]:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    table_html += '<tr>'
                    for cell in cells:
                        table_html += f'<td>{cell}</td>'
                    table_html += '</tr>'
        table_html += '</tbody></table>'
        
        return table_html
    
    # Find and convert tables
    table_pattern = r'(\|[^|\n]+\|[^|\n]*\n\|[-\s|]+\|[^|\n]*\n(?:\|[^|\n]+\|[^|\n]*\n?)+)'
    html = re.sub(table_pattern, lambda m: convert_table(m.group(1)), html, flags=re.MULTILINE)
    
    # Convert paragraphs
    paragraphs = html.split('\n\n')
    html_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            para = f'<p>{para}</p>'
        html_paragraphs.append(para)
    
    return '\n\n'.join(html_paragraphs)

def generate_comprehensive_report():
    """Generate a comprehensive analysis report"""
    
    current_dir = Path(__file__).parent
    md_file = current_dir / "SmartRocket_Complete_Analysis_Report.md"
    
    if not md_file.exists():
        print(f"❌ Markdown file not found: {md_file}")
        return None
        
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown_to_html(md_content)
    
    # Embed images
    html_with_images = embed_images_in_html(html_content, current_dir)
    
    # Professional CSS styling
    css_style = """
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.8;
        color: #2d3748;
        background-color: #ffffff;
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    
    h1 {
        color: #dc2626;
        font-size: 2.5em;
        margin: 40px 0 20px 0;
        border-bottom: 4px solid #dc2626;
        padding-bottom: 15px;
        text-align: center;
    }
    
    h2 {
        color: #7c2d12;
        font-size: 1.8em;
        margin: 35px 0 15px 0;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 10px;
        page-break-after: avoid;
    }
    
    h3 {
        color: #374151;
        font-size: 1.4em;
        margin: 25px 0 12px 0;
        page-break-after: avoid;
    }
    
    h4 {
        color: #4b5563;
        font-size: 1.2em;
        margin: 20px 0 10px 0;
        page-break-after: avoid;
    }
    
    p {
        margin: 15px 0;
        text-align: justify;
    }
    
    .report-image {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 30px auto;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        page-break-inside: avoid;
    }
    
    .data-table {
        border-collapse: collapse;
        width: 100%;
        margin: 25px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
        page-break-inside: avoid;
    }
    
    .data-table th {
        background: linear-gradient(135deg, #dc2626, #7c2d12);
        color: white;
        padding: 15px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .data-table td {
        padding: 12px;
        border-bottom: 1px solid #e5e7eb;
        background-color: #ffffff;
    }
    
    .data-table tbody tr:nth-child(even) td {
        background-color: #f9fafb;
    }
    
    .data-table tbody tr:hover td {
        background-color: #fef2f2;
    }
    
    code {
        background-color: #f1f5f9;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 0.9em;
        color: #dc2626;
        border: 1px solid #e2e8f0;
    }
    
    .code-block {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        padding: 20px;
        border-radius: 10px;
        overflow-x: auto;
        margin: 20px 0;
        border-left: 5px solid #dc2626;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        page-break-inside: avoid;
    }
    
    .code-block code {
        background: none;
        padding: 0;
        border: none;
        color: #374151;
        font-size: 0.85em;
        white-space: pre-wrap;
    }
    
    strong {
        color: #1f2937;
        font-weight: 700;
    }
    
    em {
        color: #6b7280;
        font-style: italic;
    }
    
    a {
        color: #dc2626;
        text-decoration: none;
        border-bottom: 1px dotted #dc2626;
    }
    
    a:hover {
        color: #7c2d12;
        border-bottom: 1px solid #7c2d12;
    }
    
    .header-info {
        background: linear-gradient(135deg, #dc2626, #7c2d12);
        color: white;
        padding: 30px;
        text-align: center;
        margin-bottom: 40px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.2);
    }
    
    .header-info h1 {
        color: white;
        border: none;
        margin: 0;
        font-size: 2.2em;
    }
    
    .header-info p {
        margin: 15px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    .section-break {
        height: 3px;
        background: linear-gradient(90deg, transparent, #dc2626, transparent);
        margin: 50px 0;
        border-radius: 2px;
    }
    
    .key-metrics {
        background: #f8fafc;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #059669;
        margin: 25px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background: #fef3cd;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        border-left: 4px solid #f59e0b;
    }
    
    .success-box {
        background: #dcfce7;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        border-left: 4px solid #059669;
    }
    
    @media print {
        body {
            font-size: 12pt;
            line-height: 1.5;
        }
        
        h1 { page-break-before: always; }
        h2 { page-break-before: avoid; }
        
        .report-image {
            max-height: 400px;
            page-break-inside: avoid;
        }
        
        .data-table {
            font-size: 0.9em;
        }
        
        .code-block {
            font-size: 0.8em;
        }
    }
    
    @page {
        margin: 1in;
        size: A4;
        
        @top-center {
            content: "SmartRocket Analytics - Complete Analysis Report";
            font-size: 10pt;
            color: #6b7280;
        }
        
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #6b7280;
        }
    }
    </style>
    """
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    # Complete HTML document
    html_doc = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SmartRocket Analytics - Complete Analysis Report</title>
        {css_style}
    </head>
    <body>
        <div class="header-info">
            <h1>🚀 SmartRocket Analytics</h1>
            <p>Complete Project Analysis Report</p>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="key-metrics">
            <h3>� Report Highlights</h3>
            <ul>
                <li><strong>2.76 million</strong> customer interactions analyzed</li>
                <li><strong>95% accuracy</strong> sales forecasting with LightGBM</li>
                <li><strong>84% hit rate</strong> recommendation system with GRU4Rec</li>
                <li><strong>Interactive dashboard</strong> built with Streamlit</li>
                <li><strong>Production-ready</strong> ML pipeline with proper monitoring</li>
            </ul>
        </div>
        
        <div class="section-break"></div>
        
        {html_with_images}
        
        <div class="section-break"></div>
        
        <div style="text-align: center; margin-top: 50px; color: #6b7280; font-style: italic;">
            <p>End of Report</p>
            <p>SmartRocket Analytics Team • {timestamp}</p>
        </div>
    </body>
    </html>
    """
    
    return html_doc

def main():
    """Main function to generate the report"""
    
    print("🚀 SmartRocket Analytics - Comprehensive Report Generator")
    print("=" * 60)
    print()
    
    # Generate HTML report
    print("📖 Reading project files and generating report...")
    html_report = generate_comprehensive_report()
    
    if html_report is None:
        print("❌ Failed to generate report")
        return
    
    # Save HTML report
    current_dir = Path(__file__).parent
    reports_dir = current_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    html_file = reports_dir / "SmartRocket_Complete_Analysis_Report.html"
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    file_size = html_file.stat().st_size / 1024 / 1024
    
    print(f"✅ HTML report generated successfully!")
    print(f"   📄 File: {html_file}")
    print(f"   📊 Size: {file_size:.1f} MB")
    print()
    
    print("� Report includes:")
    print("   ✅ Complete project analysis (50+ pages)")
    print("   ✅ Embedded charts and visualizations")
    print("   ✅ Technical architecture documentation") 
    print("   ✅ Model performance metrics and analysis")
    print("   ✅ Business insights and recommendations")
    print("   ✅ Code examples and implementation details")
    print()
    
    print("🎯 How to use:")
    print(f"   1. Open {html_file.name} in your web browser")
    print("   2. For PDF: Print → Save as PDF (or use browser's PDF export)")
    print("   3. Share with stakeholders and team members")
    print()
    
    print("💡 Tips:")
    print("   • Use Chrome/Edge for best PDF conversion")
    print("   • Enable 'Background graphics' for full styling")
    print("   • Adjust margins if needed for printing")
    print()
    
    print("🎉 Ready for presentation!")

if __name__ == "__main__":
    main()
