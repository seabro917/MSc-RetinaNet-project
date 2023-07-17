% Return the mat_file name of the given index. 
% Note that the index in the file name is in increasing order.
function name_of_file = return_name_of_mat_based_index(index)
myFolder = 'D:\Studying\RetinaNet_Project\AllMAT';
filePattern = fullfile(myFolder, '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
name_of_file = theFiles(index).name;
end

