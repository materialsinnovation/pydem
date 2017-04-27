.. pyDEM documentation master file, created by
   sphinx-quickstart on Tue Apr 25 10:41:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Design Exploration Method (pyDEM)
========================================

.. toctree::
   :hidden:
   :maxdepth: 1
   
   Introduction
   API
   Requirements
   GitHub
   License
   
.. image:: ../../images/pydemlogo.jpg

Inductive Design Exploration Method (IDEM)
------------------------------------------
The Inductive Design Exploration Method (IDEM) is a robust method for accounting for uncertainty propagation through multi-level problems. IDEM combines several solution-finding strategies, such as top-down (inductive) and bottom-up (deductive) design exploration, discretized ranges of parameters, with recursive function evaluation to identify robust solutions to materials design problems. This approach lends itself to developing process-structure-property-performance (PSPP) mappings of material systems, and can help identify materials of interest for future exploration.

For further references please see:

* Choi, Haejin, et al. "An inductive design exploration method for robust multiscale materials design." Journal of Mechanical Design 130.3 (2008): 031402, `<doi:10.1115/1.2829860>`_
* McDowell, David L., et al. Integrated design of multiscale, multifunctional materials and products. Butterworth-Heinemann, 2009.


pyDEM
------
Utilizing a Python-scripted framework for IDEM, the Python Design Exploration Method (pyDEM) implements a fully general IDEM instantiation that allows for the linkage of input variables, objective functions, and levels or mappings. A short introduction is covered in the examples section. Any additional code or example contributions are welcome.

Documentation
-------------
See the pyDEM examples and :doc:`API`.


Index and search
==================

* :ref:`genindex`
* :ref:`search`
