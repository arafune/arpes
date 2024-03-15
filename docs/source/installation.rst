.. _installation:

Installation
============

Some common issues in installation have been written up in the
:doc:`FAQ <faq>`.

You can install PyARPES in an editable configuration so that you can
edit it to your needs (recommended) or as a standalone package from a
package manager. In the latter case, you should put any custom code in a
separate module which you import together with PyARPES to serve your
particular analysis needs.

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

Using an installation from source is the best option if you want to
frequently change the source of PyARPES as you work. You can use code
available either from the main repository at
`GitHub <https://github.com/arafune/arpes>`.

1. **Install `rye <https://rye-up.com>`__** 
2. Clone or otherwise download the respository

.. code:: bash

   git clone https://github.com/arafune/arpes

3. Make a conda environment according to our provided specification

.. code:: bash

   cd path/to/python-arpes
   rye sync

3. Activate the environment

.. code:: bash

   activate .venv/bin/activate

4. Install PyARPES in an editable configuration


5. *Recommended:* Configure IPython kernel according to the **Barebones
   Kernel Installation** below

Additional Suggested Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install and configure standard tools like
   `Jupyter <https://jupyter.org/>`__ or `Jupyter Lab <https://jupyterlab.readthedocs.io/en/latest>`__. Notes on
   installing and configuring Jupyter based installations can be found
   in ``jupyter.md``
2. Explore the documentation and example notebooks at 
   `the documentation site <https://arpes-v4.readthedocs.io/en/daredevil/>`__.

Barebones kernel installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have Jupyter and just need to register your environment.
You can do

.. code:: bash

   pip install ipykernel
   python -m ipykernel install --user 

You can also give the kernel a different display name in Juptyer with
``python -m ipykernel install --user --display-name "My Name Here"``.

