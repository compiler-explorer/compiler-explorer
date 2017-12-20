unit output;

interface

procedure MaxArray(var X: array of Double; const Y: array of Double);

implementation

procedure MaxArray(var X: array of Double; const Y: array of Double);
var
  I: Integer;
begin
  for I := 0 to 32767 do
  begin
    if (Y[I] > X[I]) then
      X[I] := Y[I];
  end;
end;

end.

