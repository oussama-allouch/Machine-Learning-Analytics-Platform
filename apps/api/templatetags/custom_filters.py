from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Template filter to access dictionary items by key.
    Usage: {{ dict|get_item:key }}
    """
    if dictionary and isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''