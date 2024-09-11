# Generated by Django 4.1 on 2024-08-20 12:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scoring', '0002_rename_userinfo_question'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='choices',
            field=models.CharField(blank=True, default=None, max_length=300, null=True),
        ),
        migrations.AddField(
            model_name='question',
            name='question_type',
            field=models.CharField(default=None, max_length=50),
        ),
        migrations.AlterField(
            model_name='question',
            name='answer',
            field=models.CharField(default=None, max_length=300),
        ),
    ]