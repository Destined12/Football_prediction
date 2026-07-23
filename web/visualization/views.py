from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.conf import settings
import os


def figures_gallery(request):
    figures_dir = settings.RESULTS_DIR / 'figures'
    categories = {}
    category_labels = {
        'eda': 'Exploratory Data Analysis',
        'model': 'Model Evaluation',
        'explainability': 'SHAP Explainability',
    }
    for category in ['eda', 'model', 'explainability']:
        cat_dir = figures_dir / category
        if cat_dir.exists():
            pngs = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            categories[category] = {
                'label': category_labels.get(category, category.title()),
                'figures': pngs,
            }
        else:
            categories[category] = {
                'label': category_labels.get(category, category.title()),
                'figures': [],
            }

    context = {
        'categories': categories,
    }
    return render(request, 'reports/figures_gallery.html', context)


def serve_figure(request, category, filename):
    figures_dir = settings.RESULTS_DIR / 'figures' / category
    filepath = figures_dir / filename

    if not filepath.exists() or not filepath.is_file():
        raise Http404

    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        '.png': 'image/png',
        '.svg': 'image/svg+xml',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    response = HttpResponse(filepath.read_bytes(), content_type=content_type)
    response['Cache-Control'] = 'public, max-age=3600'
    return response
