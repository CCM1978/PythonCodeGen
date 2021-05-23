""" File contains all admin exports for the ViloSky app
"""
from django.contrib import admin
from django.apps import apps

# Retrieve and register all models
all_models = apps.get_models()
for model in all_models:
    try:
        admin.site.register(model)
    except admin.sites.AlreadyRegistered:
        pass
