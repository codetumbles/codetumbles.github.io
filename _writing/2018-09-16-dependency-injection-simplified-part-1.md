---
title: "Dependency Injection Simplified"
date: 2018-09-16
last_updated: 2018-09-16
categories: [design_pattern]
tags:
  - dependency-injection
  - code
  - csharp
series: dependency-injection-simplified
part: 1
series_title: "Dependency Injection Simplified"
---

Over the years, I have met a lot of fellow developers that are horrified on the mention of design patterns, calling them confounding and obscure. Contrary to this notion, I find Software architecture to be the most fun part in any software project. Recently, I found myself explaining to few acquaintances and colleagues about how their wrong interpretation of  dependency injection is making more of mess in code than before, so I decided to put together this post for them to make my life easy.

Before we start off, a ***crucial*** point to remember is that design patterns should be used only when there's a need for them and unnecessarily stuffing your code with them will yield nothing but an increasing code complexity.

Now Contrary to the common practice of defining terms at start, lets start with code scenario and name the terms and concepts as we come across them. *For sake of brevity, we are going to make classes* ***non-generic***.

### The Olden Way

Suppose your boss asks you to save user's preferences as XML file. One way of going about it would be to write ***xml Serialization*** class for it:


```csharp
class XmlSerialization
{
    /// <summary>
    /// Read Xml File and returns it as UserPreference object
    /// </summary>
    /// <param name="filePath">FileName including file path</param>
    public UserPreference ReadFile(string filePath)
    {
        using (TextReader reader = new StreamReader(filePath))
        {
            var serializer = new XmlSerializer(typeof(UserPreference));
            return (UserPreference)serializer.Deserialize(reader);
        }
    }

    /// <summary>
    /// Writes UserPreference object to Xml File
    /// </summary>
    /// <param name="obj">Object to write</param>
    /// <param name="filePath">FileName including file path</param>
    /// <param name="append">Append or Overwrite to File</param>
    public void WriteFile(UserPreference obj, string filePath, bool append = false)
    {
        using (TextWriter writer = new StreamWriter(filePath, append))
        {
            var serializer = new XmlSerializer(typeof(UserPreference));
            serializer.Serialize(writer, obj);
        }
    }
}
```



```csharp
public class UserPreference
{
    //Add Properties here
}
```



And then use xml serialization class as:


```csharp
//Instantiate UserPreference class
UserPreference Preference = new UserPreference();
XmlSerialization serializer = new XmlSerialization();
//Overwrite 'UserPrefsFile' file
serializer.WriteFile(Preference, "UserPrefsFile");
```

```csharp
//Read from 'UserPrefsFile' file
serializer.ReadFile("UserPrefsFile");
```

Now somewhere in future, the program's running fine and everyone's happy *BUT* one day your boss comes in and tells you that users are found messing around with the XML files, resulting in corrupted files and we don't want that from now on so let's start saving them as ***binary*** files. After shouting about how you are fed up with these dumb users, you would start off with writing a class for Binary Serialization:

```csharp
class BinarySerialization
{
    /// <summary>
    /// Read Binary File and returns it as UserPreference object
    /// </summary>
    /// <param name="filePath">FileName including file path</param>
    public UserPreference ReadFile(string filePath)
    {
        using (Stream stream = File.Open(filePath, FileMode.Open))
        {
            var binaryFormatter = new BinaryFormatter();
            return (UserPreference)binaryFormatter.Deserialize(stream);
        }
    }

    /// <summary>
    /// Writes UserPreference object as Binary File
    /// </summary>
    /// <param name="obj">Object to write</param>
    /// <param name="filePath">FileName including file path</param>
    /// <param name="append">Append or Overwrite to File</param>
    public void WriteFile(UserPreference obj, string filePath, bool append = false)
    {
        using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
        {
            new BinaryFormatter().Serialize(stream, obj);
        }
    }
}
```



Add **[Serializable]** data annotation on UserPreference class:
```csharp
[Serializable]
public class UserPreference
{
    //Add Properties here
}
```


 And then you would have to go on a hunt for replacing:


```csharp
XmlSerialization serializer = new XmlSerialization();
```

to 

```csharp
BinarySerialization serializer = new BinarySerialization();
```



If that isn't enough hassle, now you have to redeploy all the effected assemblies again. And if you have to make these changes in *legacy* code, then there's also the chance that the previous developer who wrote XML Serialization had a class signature like:

```csharp
class Serialization<T> where T : class
{
    public void WriteToXML(UserPreference obj, string filePath, bool append = false)
    {
        //Implementation here
    }

    public UserPreference ReadFromXML(string filePath)
    {
        //Implementation here
    }
}
```

Now you also have to change ***function names*** wherever this function is getting called in code and this can even extend to having different ***function parameters***.




### The Dependency Injection Way

Dependency injection uses ***interface*** at it's core. First you declare an interface for serialization:

```csharp
interface ISerialization
{
    void WriteFile(UserPreference obj, string filePath, bool append = false);
    UserPreference ReadFile(string filePath);
}
```



And then implement it for both XML and Binary Serialization:

```csharp
class XmlSerialization : ISerialization
{
    /// <summary>
    /// Read Xml File and returns it as UserPreference object
    /// </summary>
    /// <param name="filePath">FileName including file path</param>
    public UserPreference ReadFile(string filePath)
    {
        using (TextReader reader = new StreamReader(filePath))
        {
            var serializer = new XmlSerializer(typeof(UserPreference));
            return (UserPreference)serializer.Deserialize(reader);
        }
    }
    
    /// <summary>
    /// Writes UserPreference object to Xml File
    /// </summary>
    /// <param name="obj">Object to write</param>
    /// <param name="filePath">FileName including file path</param>
    /// <param name="append">Append or Overwrite to File</param>
    public void WriteFile(UserPreference obj, string filePath, bool append = false)
    {
        using (TextWriter writer = new StreamWriter(filePath, append))
        {
            var serializer = new XmlSerializer(typeof(UserPreference));
            serializer.Serialize(writer, obj);
        }
    }
}
```

```csharp
class BinarySerialization : ISerialization
{
    /// <summary>
    /// Read Binary File and returns it as UserPreference object
    /// </summary>
    /// <param name="filePath">FileName including file path</param>
    public UserPreference ReadFile(string filePath)
    {
        using (Stream stream = File.Open(filePath, FileMode.Open))
        {
            var binaryFormatter = new BinaryFormatter();
            return (UserPreference)binaryFormatter.Deserialize(stream);
        }
    }
    
    /// <summary>
    /// Writes UserPreference object as Binary File
    /// </summary>
    /// <param name="obj">Object to write</param>
    /// <param name="filePath">FileName including file path</param>
    /// <param name="append">Append or Overwrite to File</param>
    public void WriteFile(UserPreference obj, string filePath, bool append = false)
    {
        using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
        {
            new BinaryFormatter().Serialize(stream, obj);
        }
    }
}
```



Afterwards, we are going to make ***Serialization*** class where the real magic takes place:

```csharp
class Serialization
{
    private ISerialization _serializer;

    public Serialization() : this(new XmlSerialization())
    {
        //If constructor doesn't have serialization argument, pass XmlSerialization as default Serializer
    }

    internal Serialization(ISerialization serializer)
    {
        _serializer = serializer;
    }

    public void WriteFile(UserPreference obj, string filePath, bool append = false)
    {
        _serializer.WriteFile(obj, filePath, append);
    }

    public UserPreference ReadFile(string filePath)
    {
        return _serializer.ReadFile(filePath);
    }
}
```



***Serialization*** class have 2 constructors. One takes **ISerialization** type object as parameter, which is then used for serialization/deserialization purpose, and the other one is the default public parameter-less constructor which passes **XmlSerialization** object to the other constructor. Here instead of instantiating *serializer* in class and hence making the class dependent on *serializer*, we are ***injecting*** it as interface in class through constructor and using that for serialization. This is known as **Constructor Injection**. There are also other ways to implement dependency injection but *Constructor Injection* is the most commonly used one.
Now **Serialization** class has become independent from any particular implementation of serialization but you would still need to build and redeploy the assemblies whenever you want to change the serializer. We can solve this by making a factory class:

```csharp
class Factory
{
    private static ISerialization _serializer;
    private static object objLock = new object();

    public static ISerialization GetSerializer()
    {
        lock (objLock)
        {
            if (_serializer == null)
            {
                string className = ConfigurationManager.AppSettings["Serialization.ClassName"];

                if (String.IsNullOrEmpty(className))
                    throw new ApplicationException("Missing config Key for Serialization");

                Assembly assembly = Assembly.GetExecutingAssembly();
                _serializer = assembly.CreateInstance(className) as ISerialization;

                if (_serializer == null)
                    throw new ApplicationException(
                        string.Format("Unable to instantiate ISerialization class {0}",
                        className));
            }
            return _serializer;
        }
    }
}
```

And Adding a new key-value in *AppSettings* section of ***App.config*** file (Replace ***Namespace*** in *value* with namespace of your Serialization class):
```csharp
<appSettings>
  <add key="Serialization.ClassName" value="Namespace.XmlSerialization"/>
</appSettings>
```



This Factory class checks the ***App.config*** for serialization class name and use *Reflection* to instantiate the found class, which is then returned as ***ISerialization*** type object. This way you only have to change the value of serialization class name in ***App.config***  whenever you want to switch between **XMLSerialization** and **BinarySerialization** and spare yourself the continuous hassle of building and redeploying the assemblies.

Even though we implemented everything from scratch here but usually you would use dependency injection framework to help you with this. In the next part, I will explain how we can use **Unity**  (a popular framework for dependency injection) to achieve the same thing as we did here.
