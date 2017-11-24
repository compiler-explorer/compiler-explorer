unit output;

interface

function SumOverArray(const Input: array of Integer): Integer;

implementation

function SumOverArray(const Input: array of Integer): Integer;
var
  I: Integer;
begin
  SumOverArray := 0;

  for I := Low(Input) to High(Input) do
  begin
    SumOverArray += Input[I];
  end;
end;

end.
