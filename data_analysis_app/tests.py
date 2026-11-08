"""Tests for the CSV upload and analysis API."""

import tempfile
from pathlib import Path

from django.test import override_settings
from django.urls import reverse
from rest_framework.test import APITestCase

from utils.query_engine import Engine

from . import engines
from .models import Dataset, Message


class DatasetApiTests(APITestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.media_root = Path(self.temp_dir.name)
		self.settings_override = override_settings(MEDIA_ROOT=self.media_root)
		self.settings_override.enable()

	def tearDown(self):
		for dataset_id in Dataset.objects.values_list("id", flat=True):
			engines.discard(dataset_id)
		Dataset.objects.all().delete()
		self.settings_override.disable()
		self.temp_dir.cleanup()

	def upload_small_csv(self):
		from django.core.files.uploadedfile import SimpleUploadedFile

		file_data = (
			b"order_id,revenue,region\n"
			b"A,10,North\n"
			b"A,10,North\n"
			b"B,20,South\n"
		)
		upload = SimpleUploadedFile("sales.csv", file_data, content_type="text/csv")
		return self.client.post(reverse("dataset-list"), {"file": upload}, format="multipart")

	def test_upload_profiles_csv_and_uses_deduplicated_row_count(self):
		response = self.upload_small_csv()

		self.assertEqual(response.status_code, 201, response.data)
		self.assertEqual(response.data["row_count"], 2)
		self.assertEqual(response.data["duplicates"], 1)
		self.assertEqual(len(response.data["columns"]), 3)

	def test_header_only_csv_is_rejected_without_leaving_a_dataset(self):
		from django.core.files.uploadedfile import SimpleUploadedFile

		upload = SimpleUploadedFile("empty.csv", b"id,value\n", content_type="text/csv")
		response = self.client.post(reverse("dataset-list"), {"file": upload}, format="multipart")

		self.assertEqual(response.status_code, 400)
		self.assertIn("no data rows", response.data["detail"])
		self.assertFalse(Dataset.objects.exists())

	def test_description_question_uses_profile_without_ollama(self):
		response = self.upload_small_csv()
		dataset_id = response.data["id"]

		answer = self.client.post(
			reverse("dataset-ask", args=[dataset_id]),
			{"question": "Describe this dataset"},
			format="json",
		)

		self.assertEqual(answer.status_code, 201, answer.data)
		payload = answer.data["assistant_message"]["payload"]
		self.assertEqual(payload["kind"], "profile")
		self.assertIn("revenue", payload["profile"])

	def test_delete_does_not_remove_file_outside_chart_directory(self):
		response = self.upload_small_csv()
		dataset_id = response.data["id"]
		outside_file = self.media_root / "keep-me.txt"
		outside_file.write_text("not a chart", encoding="utf-8")
		dataset = Dataset.objects.get(pk=dataset_id)
		Message.objects.create(
			dataset=dataset,
			role=Message.ASSISTANT,
			content="test payload",
			payload={"chart_path": str(outside_file)},
		)

		deleted = self.client.delete(reverse("dataset-detail", args=[dataset_id]))

		self.assertEqual(deleted.status_code, 204)
		self.assertTrue(outside_file.exists())


class EngineCacheTests(APITestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.csv_path = Path(self.temp_dir.name) / "sales.csv"
		self.csv_path.write_text("region,revenue\nNorth,10.5\nSouth,20.25\n", encoding="utf-8")

	def tearDown(self):
		engines.discard(-1)
		self.temp_dir.cleanup()

	def fake_dataset(self):
		class FakeFile:
			path = str(self.csv_path)

		class FakeDataset:
			id = -1
			file = FakeFile()

		return FakeDataset()

	def test_using_reloads_an_engine_closed_after_lookup(self):
		dataset = self.fake_dataset()
		stale = engines.get(dataset)
		engines.discard(dataset.id)  # another request evicts it before this one takes the lock
		self.assertTrue(stale.closed)

		with engines.using(dataset) as engine:
			frame, _ = engine.run("SELECT COUNT(*) AS n FROM data")
		self.assertEqual(frame.iloc[0, 0], 2)

	def test_decimal_results_are_numeric(self):
		engine = Engine(self.csv_path)
		try:
			frame, _ = engine.run(
				'SELECT "region", CAST(SUM("revenue") AS DECIMAL(18, 2)) AS total FROM data GROUP BY 1'
			)
		finally:
			engine.close()
		self.assertEqual(str(frame["total"].dtype), "float64")


class SuggestionTests(APITestCase):
	def test_id_columns_are_skipped_but_paid_columns_are_kept(self):
		from .serializers import DatasetSerializer

		dataset = Dataset(
			name="payments.csv",
			columns=[
				{"name": "customer_id", "type": "BIGINT", "kind": "number"},
				{"name": "amount_paid", "type": "DOUBLE", "kind": "number"},
				{"name": "region", "type": "VARCHAR", "kind": "category"},
			],
		)

		suggestions = DatasetSerializer().get_suggestions(dataset)

		self.assertIn("Show total amount_paid by region in a bar chart", suggestions)


class SmallTalkTests(DatasetApiTests):
	def test_small_talk_pattern(self):
		from utils.analysis import SMALL_TALK

		for text in ["hi", "Hi!", "hello there", "hey", "thanks", "Thank you so much!", "good morning",
				"what can you do?", "help", "ok", "bye", "assalamualaikum"]:
			self.assertTrue(SMALL_TALK.match(text), text)
		for text in ["hi, what is total revenue?", "show revenue by region", "help me find top products",
				"which region has the highest profit?", "thanks, now show profit by month"]:
			self.assertFalse(SMALL_TALK.match(text), text)

	def test_greeting_gets_a_reply_without_calling_the_model(self):
		dataset_id = self.upload_small_csv().data["id"]

		answer = self.client.post(reverse("dataset-ask", args=[dataset_id]), {"question": "hi"}, format="json")

		self.assertEqual(answer.status_code, 201, answer.data)
		payload = answer.data["assistant_message"]["payload"]
		self.assertEqual(payload["kind"], "chat")
		self.assertIsNone(payload["llm_report"])
		self.assertIn("Show total revenue by region in a bar chart", payload["suggestions"])

	def test_message_without_a_query_asks_for_a_data_question_instead_of_failing(self):
		from utils.analysis import analyze

		class NoQueryLLM:
			stats = {"calls": 0}

			def chat(self, *args, **kwargs):
				self.stats["calls"] += 1
				return '{"answerable": true, "reason": "", "assumptions": [], "sql": ""}'

			def speed_report(self):
				return f"{self.stats['calls']} LLM calls"

		engine = Engine(Path(__file__).resolve().parent.parent / "data.csv")
		llm = NoQueryLLM()
		try:
			result = analyze("tell me a joke", llm, engine, "profile", {}, 0)
		finally:
			engine.close()

		self.assertEqual(result["kind"], "clarify")
		self.assertEqual(llm.stats["calls"], 1)  # no pointless "repair" retries
		self.assertTrue(result["suggestions"])
