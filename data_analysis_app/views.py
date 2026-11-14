"""REST API: upload CSVs, list them, and chat with each one."""

import logging
from pathlib import Path

from django.conf import settings
from django.db import transaction
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from utils.analysis import analyze
from utils.csv_profile import build_profile
from utils.llm_client import LLMError, make_llm
from utils.query_engine import Engine
from utils.report import format_result

from . import engines
from .models import Dataset, Message
from .serializers import AskSerializer, DatasetSerializer, MessageSerializer

logger = logging.getLogger(__name__)

NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                 "UINTEGER", "UBIGINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL")
CATEGORY_MAX_DISTINCT = 30


def describe_columns(engine):
    """Classify each column as number, date, boolean, category or text."""
    types = engine.column_types()
    text_columns = [name for name, type_name in types.items() if type_name.upper() == "VARCHAR"]
    distinct = {}
    if text_columns:
        expressions = ", ".join(
            f'approx_count_distinct("{name.replace(chr(34), chr(34) * 2)}")' for name in text_columns
        )
        distinct = dict(zip(text_columns, engine.con.execute(f"SELECT {expressions} FROM data").fetchone()))

    columns = []
    for name, type_name in types.items():
        upper = type_name.upper()
        if upper.startswith(NUMERIC_TYPES):
            kind = "number"
        elif upper.startswith(("DATE", "TIMESTAMP")):
            kind = "date"
        elif upper == "BOOLEAN":
            kind = "boolean"
        elif upper == "VARCHAR" and distinct.get(name, 0) <= CATEGORY_MAX_DISTINCT:
            kind = "category"
        else:
            kind = "text"
        columns.append({"name": name, "type": type_name, "kind": kind})
    return columns


def chart_url(chart_path):
    """Map a chart file under CHART_ROOT to the URL that serves it."""
    if not chart_path:
        return None
    path = Path(chart_path).resolve()
    root = Path(settings.CHART_ROOT).resolve()
    if root not in path.parents:
        return None
    return f"{settings.MEDIA_URL}charts/{path.relative_to(root).as_posix()}"



def build_llm():
    """Client for the backend chosen in settings (LLM_BACKEND=ollama or vllm)."""
    if settings.LLM_BACKEND == "vllm":
        return make_llm("vllm", settings.VLLM_MODEL, settings.VLLM_URL)
    return make_llm("ollama", settings.OLLAMA_MODEL)


class DatasetViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """
    GET    /api/datasets/                 list uploaded CSVs
    POST   /api/datasets/                 upload a CSV (multipart field "file")
    GET    /api/datasets/{id}/            dataset details
    DELETE /api/datasets/{id}/            delete a dataset, its file and its chat
    GET    /api/datasets/{id}/messages/   chat history
    POST   /api/datasets/{id}/ask/        ask a question: {"question": "..."}
    """

    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def create(self, request):
        upload = request.FILES.get("file")
        if upload is None:
            return Response({"detail": 'Upload a CSV file in the "file" field.'}, status=status.HTTP_400_BAD_REQUEST)
        if not upload.name.lower().endswith(".csv"):
            return Response({"detail": "Only .csv files are supported."}, status=status.HTTP_400_BAD_REQUEST)
        if upload.size > settings.CSV_MAX_UPLOAD_BYTES:
            limit_mb = settings.CSV_MAX_UPLOAD_BYTES // (1024 * 1024) #50 MB
            
            return Response({"detail": f"The file is larger than {limit_mb} MB."}, status=status.HTTP_400_BAD_REQUEST)
        if upload.size == 0:
            return Response({"detail": "The file is empty."}, status=status.HTTP_400_BAD_REQUEST)

        dataset = Dataset(name=Path(upload.name).name[:255], size_bytes=upload.size)
        dataset.file.save(upload.name, upload, save=False)
        path = dataset.file.path
        engine = None
        try:
            profile_text, missing, duplicates = build_profile(path)
            engine = Engine(path)
            dataset.profile_text = profile_text
            dataset.missing = missing
            dataset.duplicates = duplicates
            # Normal analysis uses the de-duplicated table; keep the UI row count consistent.
            dataset.row_count = engine.row_count("data")
            dataset.columns = describe_columns(engine)
            dataset.save()
            engines.put(dataset.id, engine)
        except Exception as exc:
            if engine is not None:
                engine.close()
            dataset.file.delete(save=False)
            if dataset.pk:
                dataset.delete()
            logger.info("Rejected CSV upload %s: %s", upload.name, exc)
            message = str(exc).split("\n")[0][:300]
            return Response({"detail": f"Could not read this CSV: {message}"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(self.get_serializer(dataset).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, dataset):
        engines.discard(dataset.id)
        for message in dataset.messages.all():
            chart = (message.payload or {}).get("chart_path")
            if chart:
                chart_path = Path(chart).resolve()
                chart_root = Path(settings.CHART_ROOT).resolve()
                if chart_root in chart_path.parents:
                    chart_path.unlink(missing_ok=True)
                else:
                    logger.warning("Refusing to delete chart outside chart directory: %s", chart_path)
        dataset.file.delete(save=False)
        dataset.delete()

    @action(detail=True, methods=["get"])
    def messages(self, request, pk=None):
        dataset = self.get_object()
        return Response(MessageSerializer(dataset.messages.all(), many=True).data)

    @action(detail=True, methods=["post"])
    def ask(self, request, pk=None):
        dataset = self.get_object()
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]

        user_message = Message.objects.create(dataset=dataset, role=Message.USER, content=question)
        llm = build_llm()
        response_status = status.HTTP_201_CREATED
        try:
            with engines.using(dataset) as engine:
                result = analyze(
                    question, llm, engine,
                    dataset.profile_text, dataset.missing, dataset.duplicates,
                    chart_dir=settings.CHART_ROOT,
                    on_status=lambda message: logger.info("  (%s)", message),
                )
            result["chart_url"] = chart_url(result["chart_path"])
            content = result["answer"]
        except LLMError as exc:
            result = {"kind": "error", "answer": str(exc)}
            content = str(exc)
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE
        except Exception:
            logger.exception("Question failed for dataset %s", dataset.id)
            result = {"kind": "error", "answer": "Something went wrong while answering. See the server log."}
            content = result["answer"]
            response_status = status.HTTP_500_INTERNAL_SERVER_ERROR

        # Print the same report the command-line tool shows to the runserver terminal.
        logger.info(
            "\n%s\nDATASET: %s | QUESTION: %s\n\n%s\n%s",
            "=" * 70, dataset.name, question, format_result(result), "=" * 70,
        )

        with transaction.atomic():
            assistant_message = Message.objects.create(
                dataset=dataset, role=Message.ASSISTANT, content=content, payload=result,
            )
        return Response(
            {
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(assistant_message).data,
            },
            status=response_status,
        )
