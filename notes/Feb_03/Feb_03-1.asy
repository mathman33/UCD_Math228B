if(!settings.multipleView) settings.batchView=false;
settings.tex="xelatex";
defaultfilename="Feb_03-1";
if(settings.render < 0) settings.render=4;
settings.outformat="";
settings.inlineimage=true;
settings.embed=true;
settings.toolbar=false;
viewportmargin=(2,2);

import graph;
real unit = 1cm;
unitsize(unit);

// Set the font size to match the document.
defaultpen(fontsize(10pt));

// Compute the desired paths.
path lngraph = graph(log, 0.25, 4);
path xaxis = (-.5,0) -- (4.5,0);
path yaxis = (0,-1.5) -- (0,2.5);
path yEqualsOne = (0,1) -- (4,1);

// Compute the path times of the intersection points.
real lowerisection = intersect(lngraph, xaxis)[0];
real upperisection = intersect(lngraph, yEqualsOne)[0];

// Fill the region.
fill((0,0) --
subpath(lngraph, lowerisection, upperisection)
-- (0,1) -- cycle,
0.5*gray + 0.5*white);

// Draw the paths.
draw(yEqualsOne);
draw(lngraph, L=Label("$y=\ln x$", EndPoint, align=NW));

// Draw the axes.
draw(xaxis, L=Label("$x$", EndPoint));
draw(yaxis, L=Label("$y$", EndPoint));

// Add the ticks.
real ticksize = 2pt / unit;
for (int x = 1; x <= 4; ++x)
draw((x,ticksize) -- (x,-ticksize), L=Label("$"+(string)x+"$",EndPoint));
for (int y = 1; y <= 2; ++y)
draw((ticksize,y) -- (-ticksize,y), L=Label("$" + (string)y + "$", EndPoint));

// And the final label.
label("$\mathcal{D}$", (.75,.5));
