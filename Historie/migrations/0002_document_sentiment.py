# Generated by Django 3.0.5 on 2020-04-29 16:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Historie', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='sentiment',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
    ]
