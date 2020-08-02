from django.db.backends.signals import connection_created


def configure_sqlite(sender, connection, **_):
    if connection.vendor == 'sqlite':
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')

connection_created.connect(configure_sqlite)
