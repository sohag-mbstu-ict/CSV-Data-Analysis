from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve

from data_analysis.views import index

urlpatterns = [
    path("", index, name="index"),
    path("api/", include("data_analysis_app.urls")),
    path("admin/", admin.site.urls),
    # Only generated charts are served; uploaded CSVs under media/datasets stay private.
    re_path(r"^media/charts/(?P<path>[\w\-.]+\.png)$", serve, {"document_root": settings.CHART_ROOT}),
]
