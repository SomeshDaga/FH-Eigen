function [ region, coords ] = get_region(image, bbox_size, bbox_centers)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[h, w] = size(image);

y_min = round(bbox_centers(2) - bbox_size(3));
y_max = round(bbox_centers(2) + bbox_size(4));
x_max = round(bbox_centers(1) + bbox_size(2));
x_min = round(bbox_centers(1) - bbox_size(1));
if y_max > h
    y_min = y_min - (y_max - h);
    y_max = h;
elseif y_min < 1
    y_max = y_max - y_min;
    y_min = 1;
end

if x_max > w
    x_min = x_min - (x_max - w);
    x_max = w;
elseif x_min < 1
    x_max = x_max - x_min;
    x_min = 1;
end

region = image(y_min:y_max,x_min:x_max);
coords = [x_min x_max y_min y_max];
end

