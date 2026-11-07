import re

from rest_framework import serializers

from .models import Dataset, Message

# Most to least preferred measure for the starter questions.
MEASURE_HINTS = ["revenue", "sales", "amount", "profit", "income", "total", "value", "price", "cost"]


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            "id", "name", "size_bytes", "row_count", "columns",
            "duplicates", "missing", "created_at", "suggestions",
        ]
        read_only_fields = fields

    suggestions = serializers.SerializerMethodField()

    def get_suggestions(self, dataset):
        """A few starter questions based on the dataset's columns."""
        numeric = [
            column["name"] for column in dataset.columns
            if column["kind"] == "number" and not column["name"].lower().endswith("id")
        ]
        # Prefer a money-like measure (revenue, sales, ...) over e.g. an age column.
        preferred = [
            name for hint in MEASURE_HINTS for name in numeric if re.search(hint, name, re.I)
        ]
        measure = (preferred or numeric or [None])[0]
        categories = [column["name"] for column in dataset.columns if column["kind"] == "category"]
        has_dates = any(column["kind"] == "date" for column in dataset.columns)

        questions = ["Describe this dataset"]
        if measure and categories:
            questions.append(f"Show total {measure} by {categories[0]} in a bar chart")
            if len(categories) > 1:
                questions.append(f"Which {categories[1]} has the highest average {measure}?")
        if measure and has_dates:
            questions.append(f"What is the monthly trend of total {measure}?")
        return questions[:4]


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "payload", "created_at"]
        read_only_fields = fields


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000, trim_whitespace=True)
