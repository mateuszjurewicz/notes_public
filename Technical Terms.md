# Technical Terms

Place for more general technical terms

- `FLOPS`

e.g. teraflops, it stands for floatinng point operations per second that a computer can perform.

- `GraphQL` vs `REST`

REST stands for REpresentational State Transfer. It involves having separate GET, PUT, POST and DELETE endpoints. With REST, when you initiate some change or addition to a data object via POST, you in return receive all the data pertaining to the changed object, usually in JSON form. 

In GraphQL we don't define endpoints, instead we define the graph schema. All requests, which can be **queries** (asking for some information existing in the database) or **mutations** (equivalents of PUT, POST and DELETE) go through the graph schema, which defines what is possible to do with the underlying data. This has benefits for security, transparency and flexibility. Flexbility because the client can query for e.g. only ids of all entities by adjusting its query, instead of needing a separate endpoint that only returns ids. Then again, you do define the schema by adding query and mutation resolvers, which are reminiscent of REST endpoints, but more flexible. The assumption is that the underlying data is graph-structured. The downside currently is that graphql requests may result in a large number of database queries, which is time consuming or requires custom caching or database request optimization between multiple graphql resolvers.

Source: [Arjan's graphql vs rest video](https://www.youtube.com/watch?v=7ccdWqGgHaM)