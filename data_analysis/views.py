from django.shortcuts import render


def index(request):
    """Single-page frontend: upload CSVs and chat with them through the REST API."""
    return render(request, "data_analysis/index.html")

    
