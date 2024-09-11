# Generated by Django 4.1 on 2024-08-20 07:32

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(default=None, max_length=300)),
                ('answer', models.CharField(default=None, max_length=200)),
                ('reference_file_name', models.CharField(default=None, max_length=100)),
                ('reference_section_content', models.CharField(default=None, max_length=1000)),
            ],
        ),
    ]
