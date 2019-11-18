function [ region, coords ] = get_region(image, bbox_size, bbox_centers)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[h, w] = size(image);
y_min = round(bbox_centers(2)-bbox_size(2)/2);
y_max = round(bbox_centers(2)+bbox_size(2)/2);
x_max = round(bbox_centers(1)+bbox_size(1)/2);
x_min = round(bbox_centers(1)-bbox_size(1)/2);
if y_max > h
    y_min = y_min - (y_max - h);
    y_max = h;
elseif y_min < 0
    y_max = y_max - y_min;
    y_min = 0;
end

if x_max > w
    x_min = x_min - (x_max - h);
    x_max = w;
elseif x_min < 0
    x_max = x_max - x_min;
    x_min = 0;
end

region = image(y_min:y_max,x_min:x_max);
coords = [x_min x_max y_min y_max];
end

