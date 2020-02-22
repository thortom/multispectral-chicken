myFiles = dir('*.dpt'); %gets all dpt files in struct

for k = 1:length(myFiles)
  fileName = myFiles(k).name;
  fprintf(1, 'Now reading %s\n', fileName);
  a = load(fileName);
  a(:, 1) = 10 * 1e6 ./ (a(:, 1));
  fileNameStuff = split(fileName, '.');
  newFileName = string(fileNameStuff(1)) + ".csv";
  csvwrite(newFileName, a);
end
