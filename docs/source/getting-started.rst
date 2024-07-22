Get Started with PyARPES
========================

In most case, you can start with

.. code:: python

   import arpes

Loading example data
--------------------

At this point, you should be able to load the example data, an ARPES
spectrum of the topological insulator bismuth selenide:

.. code:: python

   import arpes
   from arpes.io import load_example_data
   load_example_data()

Loading your own data
---------------------

If you have the path to a piece of data you want to load as well as the
data source it comes from (see the section on
:doc:`plugins </writing-plugins>` for more detail), you can load it with
``arpes.io.load_without_dataset``:

.. code:: python

   import arpes
   from arpes.io import load_data
   load_data('epath/to/my/data.h5', location='ALS-BL7')

What’s next?
------------

With the example data in hand, you can jump into the rest of the
examples on the site. If you’re a visual learner or are new to Jupyter
and are running into issues, have a look at the :doc:`tutorial videos </example-videos>`.
Another good place to start is on the
section for :doc:`exploration <notebooks/basic-data-exploration>` of ARPES data.
