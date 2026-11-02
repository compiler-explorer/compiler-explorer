// ilspy trunk-20261018+42118298a268ef93deda05104fb31b8e83126255
using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Security;
using System.Security.Permissions;
[assembly: CompilationRelaxations(8)]
[assembly: RuntimeCompatibility(WrapNonExceptionThrows = true)]
[assembly: Debuggable(DebuggableAttribute.DebuggingModes.IgnoreSymbolStoreSequencePoints)]
[assembly: TargetFramework(".NETCoreApp,Version=v11.0", FrameworkDisplayName = ".NET 11.0")]
[assembly: AssemblyCompany("CompilerExplorer")]
[assembly: AssemblyConfiguration("Release")]
[assembly: AssemblyFileVersion("1.0.0.0")]
[assembly: AssemblyInformationalVersion("1.0.0")]
[assembly: AssemblyProduct("CompilerExplorer")]
[assembly: AssemblyTitle("CompilerExplorer")]
[assembly: SecurityPermission(SecurityAction.RequestMinimum, SkipVerification = true)]
[assembly: AssemblyVersion("1.0.0.0")]
[module: UnverifiableCode]
[module: RefSafetyRules(11)]
internal class Program
{
	private static void Classify(int x)
	{
		switch (x)
		{
		case 1:
			Console.WriteLine("one");
			break;
		case 2:
			Console.WriteLine("two");
			break;
		case 3:
			Console.WriteLine("three");
			break;
		default:
			Console.WriteLine("other");
			break;
		}
		int value = 0;
		while (value++ <= x)
		{
			Console.WriteLine(value);
		}
		Console.WriteLine("done");
	}
	private static void Main()
	{
		Classify(2);
	}
}
