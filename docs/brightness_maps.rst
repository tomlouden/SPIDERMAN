Brightness maps
=====================================

Offset hotspot
==============

parameters:

	**la0**
		Offset of the center of the hotspot in the latitude direction (unit: Degrees)

	**lo0**
		Offset of the center of the hotspot in the longitude direction (unit: Degrees)

Notes:

Because there is a sharp contrast in flux levels between *spot* and *not spot* regions, this brightness model can have issues with quantisation, which produces unphysical "steps" in the lightcurve. This can be 

square plot:

.. figure:: images/hotspot_t_square.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

    The resulting lightcurve
