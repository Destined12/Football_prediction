from django.shortcuts import render


def dataset_detail(request):
    context = {
        'seasons': [],
    }
    return render(request, 'core/dataset_info.html', context)
