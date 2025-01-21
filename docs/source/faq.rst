Frequently Asked Questions
==========================

Igor Installation
-----------------

Using the suggested invocation I get a pip error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pip on Windows appears not to like certain archival formats. While

.. code:: bash

    pip install https://github.com/arafune/igorpy

should work on most systems, you can also clone the repository:

.. code:: bash

   git clone https://github.com/chstan/igorpy.git

And then install into your environment from inside that folder.

.. code:: bash

   (my arpes env) > echo "From inside igorpy folder"
   (my arpes env) > pip install -e .

Common Issues
-------------

I tried to upgrade a package and now things aren’t working… how do I get my old code working again?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large upgrades I recommend making a new environment until you are
sure you don’t encounter issues (500 MB disk is cheap!).

It is also helpful to keep a record of “working” configurations on
systems that you use. Different package managers have better and worse
ways of dealing with this, but you can typically recover a full
installation of complex Python software with a list of the requirements
and their versions and the version for the interpreter. As a result, I
make a point to save a copy of my full requirements

.. code:: bash

   $ pip freeze > working-dependencies-py38-date-14-11-2019.txt

You can then pip or conda install from this requirements file.

If you don’t find this satisfying, you are probably a reasonable and
sane human being. Packaging software is apparently more difficult than
it would ideally be but the situation is improving.
