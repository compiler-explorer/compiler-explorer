## Compiler Explorer and Privacy

_This is a summary of our Privacy policy, not a legal document, and might be incomplete._

_For the full Privacy policy, see `static/policies/privacy.html`, or visit https://godbolt.org/#privacy_

The main Compiler Explorer site (at https://godbolt.org/) has a cookie and privacy policy, and it's expected that any
changes to the code are in compliance with those policies. It's worth taking a look at them if you're touching any area
of the code relating to storing or logging user data.

Specifically, we must remain compliant with the EU's GDPR law, which applies to EU residents wherever they are in the
world. The main way to think about the GDPR, and privacy in general, is the user should be in control of their data. As
such we go to great pains to explain what we do with their data, and that we don't store any user-identifying
information. Of course, we _do_ (at least temporarily) store their source code, which may be precious and sensitive. We
are transparent with what we do with their data. When making short links, we send an encoding of their source code for
storage, and again we must be clear how that process works. When compiling with some Microsoft compilers we send data to
the sister site www.godbolt.ms and that data is covered by
[Microsoft's Privacy Policy](https://privacy.microsoft.com/en-US/).

Users have rights over the data they create: so in theory they could ask for any data stored on them to be removed. We
have no way of tracking data (a short link, perhaps) back to an individual user, and when I asked some experts on this
the consensus was that we're OK not to supply this. If, however, we ever have user attribution (e.g., we start having
accounts), we need to support the user being able to close their account, and/or delete any data they created (e.g.,
short links). All this makes perfect sense and would probably be done anyway, as it seems useful!

We anonymise IP addresses so there's no exact mapping back to an individual using an IP. Not that it's trivial to map an
IP to a user anyway.

We shouldn't store data forever: our web logs are set to delete after a few months.

Short URLs do turn up in the web logs: from the short URL of course one can easily extract the source code embedded in
that short URL. Users are notified of this in the privacy policy. The ultimate recourse for users concerned about this
is to not use the main Compiler Explorer but instead run their own local service, which is relatively straightforward.

### Admins

A very small group of people have administrator rights on the public Compiler Explorer. Those individuals can:

- Read the logs
- Log in to the running Compiler Explorer compilation nodes
- Access the S3 storage where caches and stored information may be kept
- Access and modify the EFS storage where compilers are stored

In short, administrators can see everything that goes on. It is expected that administrators keep this deep
responsibility in mind when performing actions on Compiler Explorer, and that they keep users' privacy at the forefront
of their minds when using their administration privileges.
