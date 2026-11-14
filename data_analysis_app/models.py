from django.db import models

class Dataset(models.Model):
    """An uploaded CSV plus the profile the language model sees instead of raw rows."""

    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    size_bytes = models.BigIntegerField(default=0)
    row_count = models.IntegerField(default=0)
    columns = models.JSONField(default=list)        # [{"name": ..., "type": ...}]
    profile_text = models.TextField()
    missing = models.JSONField(default=dict)        # {column: missing cell count}
    duplicates = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.name


class Message(models.Model):
    """One chat turn. Assistant turns keep the full analysis result in `payload`."""

    USER = "user"
    ASSISTANT = "assistant"
    ROLE_CHOICES = [(USER, "User"), (ASSISTANT, "Assistant")]

    dataset = models.ForeignKey(Dataset, related_name="messages", on_delete=models.CASCADE)
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"
