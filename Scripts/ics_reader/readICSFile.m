function x = readICSFile(icsfile)
[pathstr, name, ext] = fileparts(icsfile);
icsfile
name
ext
fid=fopen(icsfile);
c=1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end
            x.head{c}=tline;
            c=c+1;
        end
fclose(fid);
x
x.head';
x
x.filename=parsekey(x,'filename');
x.filename
temp=parsekey(x,'layout size');
x.sizes=sscanf(temp,'%d %d %d %d');

x.compfile=[pathstr '/' x.filename '.ids.gz'];
x.uncompfile=[pathstr '/' x.filename '.ids'];
gunzip(x.compfile);
if (exist(x.uncompfile)==2)
    x=loadids(x);
%     delete(x.uncompfile);
end


function xout=loadids(xin)

fid=fopen(xin.uncompfile);
data=fread(fid,inf,'float32');
xin.data=reshape(data,xin.sizes(2),xin.sizes(3),xin.sizes(4));
fclose(fid);
xout=xin;

function value=parsekey(xin,key)

value='';
for k=1:length(xin.head),
    s=strfind(xin.head{k},key);
    if s==1,
        value=xin.head{k}((2+length(key)):end);
    end
end

