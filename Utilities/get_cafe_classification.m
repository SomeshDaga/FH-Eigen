function [class_map, classes, dataset] = get_cafe_classification(input_img_name, dataset)
%GET_CAFE_CLASSIFICATION Summary of this function goes here
%   Detailed explanation goes here
num_imgs = length(dataset);
class_map = containers.Map;
classes = [];
class_idx = 1;
for i=1:num_imgs
    % Check the class of the image
    img_name = string(dataset(i).file);
    
    % Exclude current image from classification
    if string(img_name) == string(input_img_name)
        continue
    end
    
    [tokens, ~] = regexp(img_name, '_(\w{1,2})\d{1}_aligned', 'tokens', 'match');
    class_tag = tokens{1}{1};
    if ~isKey(class_map, class_tag)
        class_map(class_tag) = class_idx;
        class.tag = class_tag;
        class.files = [];
        class.idxs = [];
        classes = [classes class];
        class_idx = class_idx + 1;
    end
    classes(class_map(class_tag)).files = [classes(class_map(class_tag)).files img_name];
    classes(class_map(class_tag)).idxs = [classes(class_map(class_tag)).idxs i];
end

% Return indexes of the other images in the same class
[tokens, ~] = regexp(string(input_img_name), '_(\w{1,2})\d{1}_aligned', 'tokens', 'match');
input_class_tag = tokens{1}{1};
dataset = dataset(classes(class_map(input_class_tag)).idxs);

end

