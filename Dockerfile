FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY router ./router
RUN pip install --no-cache-dir .
COPY config.yaml ./
COPY data ./data
ENV ROUTER_CONFIG=/app/config.yaml
EXPOSE 8000
CMD ["uvicorn", "router.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
