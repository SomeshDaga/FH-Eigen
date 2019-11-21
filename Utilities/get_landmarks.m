function [landmarks] = get_landmarks(file_path, is_file)
%GET_LANDMARKS Returns face landmarks for a given image
%   Makes an API call to Face++ to analyze and return facial landmarks

if nargin < 2
    is_file = true;
end

url = 'https://api-us.faceplusplus.com/facepp/v3/detect';

% These keys are used to authenticate requests to the Face++ platform
% (https://www.faceplusplus.com/)
% They should really be passed in through function parameters
% for security reasons. For ease of use, will do this at a later time
api_key = '43s5IyZ08GWg0mmswdGPL5BU6n3cDQcH';
api_secret = '4XuSk7K_VEfjhl3YDyuhHZ4M9FA9R0Jh';

return_landmark = 1; % This sets the number of landmarks to return
                     %  0 - No landmarks
                     %  1 - 83 landmarks
                     %  2 - 106 landmarks
if is_file
    % Read the byte stream in from the image file
    f = fopen(file_path);
    data = fread(f,Inf,'*uint8');
    size(data)
    fclose(f);
else
    data = imencode(file_path);
end

resp = urlreadpost(url,{'api_key', api_key,...
                        'api_secret', api_secret,...
                        'image_file', data,...                   
                        'return_landmark', int2str(return_landmark)});
                    
% Parse the response into a MATLAB struct
landmarks = jsondecode(resp);
pause(0.5);
if ~is_file
    imshow(file_path)
    hold on
    fields = fieldnames(landmarks.faces.landmark);
    for i=1:length(fields)
        feature = landmarks.faces.landmark.(fields{i});
        plot(feature.x,feature.y,'.r');
    end
    hold off
    waitforbuttonpress
end
                    
end

