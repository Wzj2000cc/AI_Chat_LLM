from django.db import models


# Create your models here.
class Question(models.Model):

    def __str__(self):
        return f"{self.question}"

    question = models.CharField(default=None, null=False, blank=False, max_length=300)
    question_type = models.CharField(default=None, null=False, blank=False, max_length=50)
    choices = models.CharField(default="", null=True, blank=True, max_length=300)
    answer = models.CharField(default=None, null=False, blank=False, max_length=300)
    LLM_output = models.CharField(default="", null=True, blank=True, max_length=300)
    reference_file_name = models.CharField(default=None, null=False, blank=False, max_length=100)
    reference_section_content = models.CharField(default=None, null=False, blank=False, max_length=1000)
