Developer Guide
===============

Topics
------

Installing an editable copy of PyARPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install `uv <https://docs.astral.sh/uv/guides/projects/>` to make an isolated environment for development.
2. Clone the respository

.. code:: bash

   git clone https://gitlab.com/arafune/arpes

3. Make virtual environment in the cloned directory, and activate it.
4. Install libraries to develop PyARPES with

   ``uv sync``.

Tests (with coverage information)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running Tests
^^^^^^^^^^^^^

.. code:: bash

   pytest -vv --conv=arpes --cov-report=html tests/

finally, you can view results at htmlcov/index.html

When to write tests
^^^^^^^^^^^^^^^^^^^

If you are adding a new feature, please consider adding a few unit
tests. Additionally, all bug fixes should come with a regression test if
they do not require a very heavy piece of fixture data to support them.

To write a test that consumes data from disk using the standard PyARPES
loading conventions, fixtures are available in ``tests/conftest.py``.
The tests extent in ``test_basic_data_loading.py`` illustrate using
these fixtures.

Contributing Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Updating existing documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update existing documentation you can simply modify the appropriate
files. You should not need to rebuild the documentation for your changes
to take effect, but there is no harm is doing so.

Rebuilding the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To rebuild the documentation you will need to have both
`sphinx <http://www.sphinx-doc.org/en/master/>`__ and
`pandoc <https://pandoc.org/>`__ installed. Then from the directory that
contains the ``setup.py`` file

1. Refresh Sphinx sources with ``sphinx-apidoc``:
   ``python -m sphinx.apidoc --separate -d 3 --tocfile toc -o source arpes --force``
2. Build Sphinx documentation to ReStructuredText:
   ``make clean && make rst``
3. Convert ReStructuredText to Markdown: ``./source/pandoc_convert.py``
4. Run ``docsify`` to verify changes: ``docsify serve ./docs``
5. As desired publish to docs site by pushing updated documentation

**Note** Sometimes ``sphinx-doc`` has trouble converting modules to
ReStructured Text.versioning This typically manifests with a
``KeyError`` in ``docutils``. This occurs when the docstrings do not
conform to the standard for ReStructuredText. The most common problem
encountered is due to bare hyperlinks, which are incompatible with the
*unique* hyperlink format in RST.

Style
~~~~~

We strongly recommend to use "black" for formatting, and request "ruff" and "mypy" free coding.
And if your code scans well and doesn’t use too many short variable names there’s no issue.
