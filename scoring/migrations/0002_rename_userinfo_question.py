# Generated by Django 4.1 on 2024-08-20 07:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('scoring', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='UserInfo',
            new_name='Question',
        ),
    ]
