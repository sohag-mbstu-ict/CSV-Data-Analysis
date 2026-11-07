from django.contrib import admin

from .models import Dataset, Message


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ["name", "row_count", "size_bytes", "created_at"]
    readonly_fields = ["created_at"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["dataset", "role", "content", "created_at"]
    list_filter = ["role", "dataset"]
