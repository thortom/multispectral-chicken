function writeICSFile(data,icsfile)

[pathstr, name, ext] = fileparts(icsfile);


fid=fopen(icsfile,'w');
fprintf(fid,'ics_version 1.0\r\n');
fprintf(fid,'filename %s\r\n',name);
fprintf(fid,'layout parameters 4\r\n');
fprintf(fid,'layout order bits x y z\r\n');
fprintf(fid,'layout sizes %d %d %d %d\r\n',32,size(data,1),size(data,2),size(data,3));
fprintf(fid,'layout coordinates video\r\n');
fprintf(fid,'layout significant_bits 32\r\n');
fprintf(fid,'representation format float\r\n');
fprintf(fid,'representation sign signed\r\n');
fprintf(fid,'representation byte_order 1 2 3 4\r\n');
fprintf(fid,'representation	SCIL_TYPE	g3d\r\n');
fprintf(fid,'SENSOR_ID 0\r\n');
fprintf(fid,'image_channels 1\r\n');
fclose(fid)

writeids(data,[pathstr '\' name '.ids'])

function writeids(data,idsfile)
fid=fopen(idsfile,'w');
fwrite(fid,data,'float32');
fclose(fid);
gzip(idsfile);
delete(idsfile);