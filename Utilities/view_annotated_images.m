function [] = view_annotated_images(landmarks_file)
%VIEW_ANNOTATED_IMAGES Display annotated images from a landmarks file

contents = load(landmarks_file);
data = contents.data;
fields = fieldnames(data(1).faces.landmark);
for i=1:length(data)
    img = imread(data(i).file);
    
    % Only show the image if required
    imshow(img)
    hold on
    for j=1:numel(fields)
        % Get coordinates of feature
        feature = data(i).faces.landmark.(fields{j});
        
        % Plot features
        plot(feature.x, feature.y, '.r');
    end
    hold off
    waitforbuttonpress
end

end

