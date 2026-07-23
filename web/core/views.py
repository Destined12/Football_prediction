from django.shortcuts import render


def home(request):
    return render(request, 'core/home.html')


def about(request):
    return render(request, 'core/about.html')


def model_info(request):
    return render(request, 'core/model_info.html')


def dataset_info(request):
    return render(request, 'core/dataset_info.html')


def methodology(request):
    return render(request, 'core/methodology.html')
