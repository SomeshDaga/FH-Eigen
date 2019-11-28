function [confidence] = get_face_similarity(face_img_1,face_img_2)
%GET_FACE_SIMILARITY Summary of this function goes here
%   Detailed explanation goes here

url = 'https://api-us.faceplusplus.com/facepp/v3/compare';
api_key = '43s5IyZ08GWg0mmswdGPL5BU6n3cDQcH';
api_secret = '4XuSk7K_VEfjhl3YDyuhHZ4M9FA9R0Jh';

face_1 = imencode(face_img_1);
face_2 = imencode(face_img_2);

resp = urlreadpost(url,{'api_key', api_key,...
                        'api_secret', api_secret,...
                        'image_file1', face_1,...                   
                        'image_file2', face_2});
resp = jsondecode(resp);
pause(0.5);
confidence = resp.confidence;
end

