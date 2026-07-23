from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.conf import settings
import io
import os
import pandas as pd


REPORT_NAMES = {
    'evaluation_metrics': 'Evaluation Metrics',
    'classification_report': 'Classification Report',
    'selected_features': 'Selected Features',
    'feature_importance': 'Feature Importance Rankings',
    'feature_selection_eval': 'Feature Selection Evaluation',
    'feature_count_eval': 'Feature Count Evaluation',
}


def index(request):
    tables_dir = settings.RESULTS_DIR / 'tables'
    report_types = []
    for key, name in REPORT_NAMES.items():
        csv_path = tables_dir / f'{key}.csv'
        if csv_path.exists():
            report_types.append({'name': name, 'key': key})
    return render(request, 'reports/reports.html', {'report_types': report_types})


def download_report(request, report_type, fmt):
    if report_type not in REPORT_NAMES:
        raise Http404
    if fmt not in ('csv', 'excel', 'pdf'):
        raise Http404

    csv_path = settings.RESULTS_DIR / 'tables' / f'{report_type}.csv'
    if not csv_path.exists():
        raise Http404

    display_name = REPORT_NAMES[report_type]

    if fmt == 'csv':
        return _serve_csv(csv_path, report_type)
    elif fmt == 'excel':
        return _serve_excel(csv_path, report_type, display_name)
    elif fmt == 'pdf':
        return _serve_pdf(csv_path, report_type, display_name)


def _serve_csv(csv_path, report_type):
    response = HttpResponse(csv_path.read_bytes(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{report_type}.csv"'
    return response


def _serve_excel(csv_path, report_type, display_name):
    df = pd.read_csv(csv_path)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        ws = writer.sheets['Data']
        for i, col in enumerate(df.columns, 1):
            max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
            ws.column_dimensions[chr(64 + i) if i <= 26 else 'A'].width = min(max_len, 40)

    buffer.seek(0)
    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename="{report_type}.xlsx"'
    return response


def _serve_pdf(csv_path, report_type, display_name):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    df = pd.read_csv(csv_path)
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=40, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=16, spaceAfter=10)
    elements.append(Paragraph(display_name, title_style))
    elements.append(Spacer(1, 10))

    headers = [str(c) for c in df.columns]
    data = [headers]
    for _, row in df.iterrows():
        data.append([str(v)[:40] for v in row])

    col_count = len(headers)
    available_width = landscape(A4)[0] - 80
    col_width = available_width / col_count

    table = Table(data, colWidths=[col_width] * col_count)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#334155')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#0f172a'), colors.HexColor('#1e293b')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{report_type}.pdf"'
    return response
