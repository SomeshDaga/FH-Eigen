function [] = generate_dataset_landmarks(dataset_path,img_type,output_file)
%GENERATE_DATASET_LANDMARKS Generates a MAT file containing landmark for
%                           all images contained in the path

% Get the list of images in the dataset
images = dir( fullfile(dataset_path, img_type) );

% Create the MAT file
data = [];
save(output_file, 'data')

% Initialize the MAT file for writing
m = matfile(output_file,'Writable',true);

% Iterate through images in dataset and generate landmarks
% Add generated landmarks to MAT file
for i=1:length(images)
    file = fullfile(dataset_path,images(i).name);
    landmarks = get_landmarks(file);
    landmarks.file = file;
    m.data = [m.data, landmarks];
end

end

